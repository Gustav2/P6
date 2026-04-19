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

from config import SIM_DURATION_S, SAT_ORBITAL_VELOCITY_MS


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

    # Lookup table: sat_id → (x_m, y_m) from channel stats — the satellite's
    # RT proxy position at the *start* of its slot.  Orbital motion is added
    # per-frame so the marker moves smoothly within each slot.
    _sat_pos_lookup = {s["sat_id"]: (s["sat_x_m"], s["sat_y_m"])
                       for s in channel_stats
                       if "sat_x_m" in s and "sat_y_m" in s}

    def _sat_pos_at(t: float):
        """Return (x_m, y_m) of the serving satellite at time t.

        Within each handover slot the satellite moves at orbital velocity
        (x-direction only, matching the NS-3 ConstantVelocityMobilityModel).
        At a handover boundary the position jumps to the new satellite's
        starting location.  Returns None when channel_stats is unavailable.
        """
        for slot in handover_schedule:
            if slot["t_start"] <= t <= slot["t_end"]:
                sid = slot["sat_id"]
                if sid in _sat_pos_lookup:
                    x0, y0 = _sat_pos_lookup[sid]
                    return (x0 + SAT_ORBITAL_VELOCITY_MS * (t - slot["t_start"]),
                            y0)
        return None

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

        # Client-area boundary
        circle = plt.Circle((0, 0), 500.0, fill=False,
                             edgecolor="#666", linestyle="--",
                             linewidth=0.8, alpha=0.7, zorder=1)
        ax.add_patch(circle)

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

        # Satellite ground-projection indicator.
        # Use the *serving* satellite's position from channel_stats so the
        # marker jumps to the correct satellite at each handover, rather than
        # following the single NS-3 proxy node's continuous trajectory.
        sat_id, elev = _serving_sat_at(t_s[f], handover_schedule)
        pos = _sat_pos_at(t_s[f])
        if pos is not None:
            sx_m, sy_m = pos
        else:
            sx_m, sy_m, _ = sat_xyz[f]    # fallback: NS-3 node position
        sx_clamped = max(-half_extent_m * 0.95,
                         min(half_extent_m * 0.95, sx_m / 1000.0))
        sy_clamped = max(-half_extent_m * 0.95,
                         min(half_extent_m * 0.95, sy_m / 1000.0))
        ax.scatter([sx_clamped], [sy_clamped], marker="*", s=250,
                   color="#d62728", edgecolors="white", linewidths=1.2,
                   zorder=6, label="serving satellite")

        # HUD — sat_id and elev already set by the satellite marker block above
        ho_count = sum(1 for s in handover_schedule if s["t_start"] <= t_s[f]
                       and s.get("interruption_ms", 0) > 0)
        hud = (f"t = {t_s[f]:5.1f} s    "
               f"serving sat{sat_id}  elev {elev:4.1f}°    "
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
        ax.set_title("NTN Client Mobility — Munich Scene", fontsize=12, weight="bold")

        ax.legend(loc="lower right", fontsize=9, framealpha=0.92,
                  edgecolor="#888", title="UE type", title_fontsize=8)

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
