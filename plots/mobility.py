"""
plots/mobility.py — Client-mobility MP4 video over the Munich city scene.

Composes a per-frame figure from:
  * A pre-rendered nadir view of the Munich scene (background)
  * Scatter markers for every UE (stationary, pedestrian, vehicular)
  * Short motion tails for moving UEs (last N frames)
  * Current serving-satellite indicator and handover HUD

Writes the frames to an MP4 via imageio (which bundles its own ffmpeg
binary, so no system-level ffmpeg install is required).
"""

import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as mlines

from config import (
    SIM_DURATION_S,
    RT_UE_POSITION,
    SAT_HEIGHT_M,
    SAT_ORBITAL_VELOCITY_MS,
    SAT_HANDOVER_ELEVATION_DEG,
)

# Earth radius for slant-range geometry (matches sim/ns3.py).
_EARTH_RADIUS_M = 6_371_000.0


_STATIONARY_STYLE = dict(color="#1f77b4", marker="s", s=30,
                          edgecolors="white", linewidths=0.6,
                          label="stationary (30)")
_PEDESTRIAN_STYLE = dict(color="#2ca02c", marker="o", s=45,
                          edgecolors="white", linewidths=0.6,
                          label="pedestrian (10)")
_VEHICULAR_STYLE  = dict(color="#ff7f0e", marker="^", s=70,
                          edgecolors="white", linewidths=0.7,
                          label="vehicular (10)")

# Motion tail length (frames)
_TAIL_FRAMES = 8


def _serving_sat_at(t: float, handover_schedule: list):
    """Return (sat_id, elev_deg) for whichever slot contains time `t`."""
    for slot in handover_schedule:
        if slot["t_start"] <= t <= slot["t_end"]:
            return slot["sat_id"], slot.get("elev_deg", float("nan"))
    # Fallback — if t is past the last slot, use the final slot
    if handover_schedule:
        last = handover_schedule[-1]
        return last["sat_id"], last.get("elev_deg", float("nan"))
    return -1, float("nan")


def _active_slot(t: float, handover_schedule: list):
    """Return the slot containing time `t`, or the last slot as fallback."""
    for slot in handover_schedule:
        if slot["t_start"] <= t <= slot["t_end"]:
            return slot
    return handover_schedule[-1] if handover_schedule else None


def _slant_range_m(height_m: float, elev_deg: float) -> float:
    """Spherical-Earth slant range [m]. Matches sim/ns3.py._slant_range_m."""
    e = math.radians(elev_deg)
    return (math.sqrt((_EARTH_RADIUS_M + height_m) ** 2
                      - (_EARTH_RADIUS_M * math.cos(e)) ** 2)
            - _EARTH_RADIUS_M * math.sin(e))


def _azel_to_xy(az_rad: float, elev_deg: float, half_extent_m: float,
                r_max_frac: float = 0.8) -> tuple:
    """
    Sky-plot projection: elevation drives radius (90° = centre, 0° = horizon),
    azimuth drives angle.  Scaled so the horizon ring sits inside the UE
    placement circle.
    """
    r_frac = max(0.0, min(1.0, (90.0 - elev_deg) / 90.0))
    r = r_frac * r_max_frac * half_extent_m
    return r * math.sin(az_rad), r * math.cos(az_rad)


def _serving_elev_at(t: float, slot: dict, peak_elev_deg: float) -> float:
    """
    Interpolate the serving-satellite elevation across its slot using the
    same d(elev)/dt physics that drove the NS-3 slot-duration weighting
    (sim/ns3.py:488–491).  Slot midpoint maps to the RT snapshot peak;
    elevation rises slightly into the slot and falls toward the handover
    threshold by slot end.
    """
    t_start = slot["t_start"]
    t_end   = slot["t_end"]
    t_mid   = 0.5 * (t_start + t_end)
    peak    = max(peak_elev_deg, 1.0)

    # rad/s angular elevation rate at peak; assumed roughly constant across
    # the slot (a good first-order approximation for a zenith pass).
    peak_rad  = math.radians(peak)
    slant_m   = _slant_range_m(SAT_HEIGHT_M, peak)
    rate_rads = SAT_ORBITAL_VELOCITY_MS * math.sin(peak_rad) / slant_m
    rate_deg  = math.degrees(rate_rads)

    elev = peak - rate_deg * (t - t_mid)
    return max(float(SAT_HANDOVER_ELEVATION_DEG), min(90.0, elev))


def render_mobility_video(position_trace: dict,
                           channel_stats: list,
                           handover_schedule: list,
                           bg_image_path: str = "output/.mobility_scene_bg.png",
                           out: str = "output/ntn_mobility.mp4",
                           fps: int = 10,
                           half_extent_m: float = 600.0) -> str:
    """
    Render the client-mobility video from a single NS-3 position trace.

    Parameters
    ----------
    position_trace : dict
        Output from run_ns3() when sample_positions=True.  Must contain:
          t_s, phones, satellite, num_stationary, num_pedestrian, num_vehicular.
    channel_stats : list  Per-sat channel stats (used for HUD labels only).
    handover_schedule : list  Slot list with sat_id, t_start, t_end, elev_deg.
    bg_image_path : str  Cached Munich nadir render.
    out           : str  Output .mp4 path.
    fps           : int  Frames per second (must match 1 / position_trace['dt_s']
                         for real-time playback).
    half_extent_m : float  Half-width of the plotted area, metres.

    Returns
    -------
    str  Path of the saved video, or an empty string if imageio is missing.
    """
    if not position_trace:
        print("[Mobility]  No position_trace — skipping video.")
        return ""

    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[Mobility]  imageio not installed — skipping video.")
        print("            pip install imageio imageio-ffmpeg")
        return ""

    t_s      = position_trace["t_s"]
    phones   = position_trace["phones"]       # list of [(x,y,z), ...] per frame
    sat_xyz  = position_trace["satellite"]    # list of (x,y,z) per frame — fallback
    n_stat   = position_trace.get("num_stationary", 30)
    n_ped    = position_trace.get("num_pedestrian", 10)

    # Sky-plot overlay for satellites.  Plotting raw (sat_x_m, sat_y_m)
    # can't work: RT proxies live at 550 km altitude with horizontal offsets
    # of 200–785 km, which overflows the 1.2 km urban scene plot.  Instead
    # we project (azimuth, elevation) onto the axes: azimuth → angle,
    # (90° − elevation)/90° → radius.  Azimuth is derived from the snapshot
    # geometry relative to the RT UE anchor.
    _ux, _uy, _ = RT_UE_POSITION
    _all_sats = [
        (s["sat_id"],
         math.atan2(s["sat_x_m"] - _ux, s["sat_y_m"] - _uy),
         s.get("elevation_deg", 0.0))
        for s in channel_stats
        if "sat_x_m" in s and "sat_y_m" in s
    ]
    _peak_by_id = {sid: (az, el) for sid, az, el in _all_sats}

    n_frames = len(t_s)
    if n_frames == 0:
        print("[Mobility]  Empty position_trace — skipping video.")
        return ""

    # Preload background (optional — if missing we just use a plain background)
    bg_img = None
    if os.path.exists(bg_image_path):
        try:
            bg_img = mpimg.imread(bg_image_path)
        except Exception as e:
            print(f"[Mobility]  Could not read {bg_image_path}: {e}")

    # Precompute split indices for the three UE classes
    stat_idx = list(range(0, n_stat))
    ped_idx  = list(range(n_stat, n_stat + n_ped))
    veh_idx  = list(range(n_stat + n_ped, len(phones[0]) if phones else n_stat + n_ped))

    # Build per-class XY arrays per frame: xy[frame] -> (x, y)
    def _xy(frame_idx: int, idx_list: list):
        frame = phones[frame_idx]
        xs = [frame[i][0] for i in idx_list]
        ys = [frame[i][1] for i in idx_list]
        return xs, ys

    # Initialize the video writer.  quality=8 is a reasonable trade-off
    # between file size and visual quality for libx264.
    try:
        writer = imageio.get_writer(out, fps=fps, codec="libx264",
                                    quality=8, macro_block_size=1)
    except Exception as e:
        print(f"[Mobility]  imageio writer failed ({e}) — falling back to GIF.")
        out = out.rsplit(".", 1)[0] + ".gif"
        writer = imageio.get_writer(out, fps=fps)

    # Reuse one figure across frames to avoid re-allocating matplotlib state.
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    fig.patch.set_facecolor("#f8f9fa")
    ext = [-half_extent_m, half_extent_m, -half_extent_m, half_extent_m]

    # Handover flash: for 0.5 s after each handover the satellite star pulses
    # (larger, gold) so the event is visually obvious even without reading HUD.
    _FLASH_S = 0.5
    _ho_starts = sorted(s["t_start"] for s in handover_schedule
                        if s.get("interruption_ms", 0) > 0)

    # Legend proxies for the dashed circles (circle patches don't auto-appear
    # in the legend, so we add dashed-line proxy handles).
    _R_MAX_FRAC = 0.8
    _HORIZON_R_M = _R_MAX_FRAC * half_extent_m
    _circle_proxy = mlines.Line2D([], [], color="#555", linestyle="--",
                                   linewidth=0.9, label="UE placement area (500 m)")
    _horizon_proxy = mlines.Line2D([], [], color="#888", linestyle=":",
                                    linewidth=0.9,
                                    label=f"sky horizon (elev 0°, r={_HORIZON_R_M:.0f} m)")

    print(f"[Mobility]  Rendering {n_frames} frames at {fps} fps -> {out}")
    log_every = max(1, n_frames // 10)

    for f in range(n_frames):
        ax.clear()
        ax.set_facecolor("#dfeaf2")

        # Background: Munich scene, darkened for contrast with markers
        if bg_img is not None:
            ax.imshow(bg_img, extent=ext, origin="upper",
                      interpolation="bilinear", alpha=0.85, zorder=0)
        else:
            ax.add_patch(plt.Rectangle((-half_extent_m, -half_extent_m),
                                        2 * half_extent_m, 2 * half_extent_m,
                                        facecolor="#dfeaf2",
                                        edgecolor="none", zorder=0))

        # Client-area boundary (UE placement circle, fixed at 500 m)
        circle = plt.Circle((0, 0), 500.0, fill=False,
                             edgecolor="#666", linestyle="--",
                             linewidth=0.8, alpha=0.7, zorder=1)
        ax.add_patch(circle)

        # Sky-plot horizon ring (radius where satellite elev = 0°).
        horizon = plt.Circle((0, 0), _HORIZON_R_M, fill=False,
                              edgecolor="#888", linestyle=":",
                              linewidth=0.8, alpha=0.6, zorder=1)
        ax.add_patch(horizon)

        # Motion tails for moving UEs (last _TAIL_FRAMES frames)
        tail_start = max(0, f - _TAIL_FRAMES)
        for tail_f in range(tail_start, f):
            alpha = 0.06 + 0.6 * (tail_f - tail_start) / max(1, _TAIL_FRAMES)
            xs, ys = _xy(tail_f, ped_idx + veh_idx)
            ax.scatter(xs, ys, s=8, color="#2ca02c", alpha=alpha * 0.5,
                       edgecolors="none", zorder=2)

        # Current-frame markers
        xs, ys = _xy(f, stat_idx)
        ax.scatter(xs, ys, zorder=3, **_STATIONARY_STYLE)
        xs, ys = _xy(f, ped_idx)
        ax.scatter(xs, ys, zorder=4, **_PEDESTRIAN_STYLE)
        xs, ys = _xy(f, veh_idx)
        ax.scatter(xs, ys, zorder=5, **_VEHICULAR_STYLE)

        # Sky-plot: all constellation satellites as grey stars at their
        # snapshot (azimuth, peak-elevation), and the serving satellite as
        # a red/gold star whose elevation sweeps within its slot.
        sat_id, _peak_elev = _serving_sat_at(t_s[f], handover_schedule)
        slot = _active_slot(t_s[f], handover_schedule)

        non_xy = [_azel_to_xy(az, el, half_extent_m, _R_MAX_FRAC)
                  for sid, az, el in _all_sats if sid != sat_id]
        if non_xy:
            non_xs, non_ys = zip(*non_xy)
            ax.scatter(non_xs, non_ys, marker="*",
                       s=80, color="#aaaaaa", edgecolors="#777", linewidths=0.6,
                       zorder=6, label="other satellites")

        if sat_id in _peak_by_id and slot is not None:
            az_s, peak_s = _peak_by_id[sat_id]
            elev_now = _serving_elev_at(t_s[f], slot, peak_s)
            sx, sy = _azel_to_xy(az_s, elev_now, half_extent_m, _R_MAX_FRAC)
        else:
            sx, sy = 0.0, 0.0
            elev_now = _peak_elev

        time_since_ho = min((t_s[f] - ts for ts in _ho_starts if ts <= t_s[f]),
                            default=float("inf"))
        is_flash = time_since_ho < _FLASH_S
        ax.scatter([sx], [sy], marker="*",
                   s=500 if is_flash else 300,
                   color="#ffa500" if is_flash else "#d62728",
                   edgecolors="white",
                   linewidths=2.0 if is_flash else 1.2,
                   zorder=7, label="serving satellite")

        # HUD — uses the animated elevation so the readout tracks the marker.
        ho_count = sum(1 for s in handover_schedule if s["t_start"] <= t_s[f]
                       and s.get("interruption_ms", 0) > 0)
        hud = (f"t = {t_s[f]:5.1f} s    "
               f"serving sat{sat_id}  elev {elev_now:4.1f}°    "
               f"handovers: {ho_count}")
        ax.text(0.02, 0.98, hud,
                transform=ax.transAxes, fontsize=11, weight="bold",
                color="#111", va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.80, edgecolor="#888",
                          boxstyle="round,pad=0.3"))

        ax.set_xlim([-half_extent_m, half_extent_m])
        ax.set_ylim([-half_extent_m, half_extent_m])
        ax.set_xlabel("x [m]", fontsize=9)
        ax.set_ylabel("y [m]", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("NTN Client Mobility — Munich Scene  ·  "
                     "Satellites in sky-plot overlay (radius = zenith angle)",
                     fontsize=11, weight="bold")

        handles, labels = ax.get_legend_handles_labels()
        handles.extend([_circle_proxy, _horizon_proxy])
        labels.extend(["UE placement area (500 m)",
                       f"sky horizon (elev 0°, r={_HORIZON_R_M:.0f} m)"])
        ax.legend(handles=handles, labels=labels,
                  loc="lower right", fontsize=9, framealpha=0.92,
                  edgecolor="#888", title="Legend", title_fontsize=8)

        # Draw to canvas and extract RGB array for the writer.
        # matplotlib ≥ 3.8 removed tostring_rgb(); use buffer_rgba() + drop alpha.
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgba = np.asarray(fig.canvas.buffer_rgba()).reshape(h, w, 4)
        frame = rgba[:, :, :3]
        writer.append_data(frame)

        if (f + 1) % log_every == 0 or f == n_frames - 1:
            print(f"  [Mobility]  frame {f + 1}/{n_frames}  (t = {t_s[f]:.1f} s)")

    writer.close()
    plt.close(fig)
    print(f"[Mobility]  Saved -> {out}  ({n_frames} frames @ {fps} fps)")
    return out
