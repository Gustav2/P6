"""
rt_sim.py — Part 1: Sionna RT ray tracing  (runs before NS-3)
==============================================================
Models the 5G-NTN uplink from a street-level UE (phone) placed in the
Munich scene to a constellation of LEO satellites passing overhead.

Because the actual orbital altitude (550 km) is far outside the Sionna RT
scene bounds (~100 m), each satellite is represented by a proxy transmitter
placed RT_SAT_SCENE_HEIGHT_M above the scene.  This produces the correct
near-vertical incidence geometry for extracting urban multipath statistics
(delay spread, shadow fading, angle-of-arrival spread) which are then fed
into the NS-3 link budget.  The true free-space path loss over 550 km is
computed analytically in ntn_ns3.py.

Multi-satellite pass
---------------------
NUM_SATELLITES proxy transmitters are placed at angular offsets
(SAT_SPACING_DEG) around the scene to represent successive satellites
in the constellation.  For each, ray tracing is performed independently
and per-satellite channel statistics are returned.

Multiple representative receivers are sampled in the scene:
  - UEs at RT_UE_SAMPLE_POSITIONS  — service link (sat → UE)

Outputs
-------
  output/ntn_rt_paths_sat<N>.png   — scene render with paths for satellite N
  output/ntn_rt_radiomap.png       — composite radio map (all satellites)

Returns
-------
  channel_stats : list[dict]
      One dict per satellite (in order of decreasing elevation angle).
      Service-link keys (sat → UE):
        sat_id                int   satellite index (0-based)
        elevation_deg         float UE elevation angle [deg]
        mean_path_gain_db     float mean |h| over sampled UEs [dB]
        mean_path_gain_p10_db float 10th-percentile gain over sampled UEs [dB]
        delay_spread_ns       float RMS delay spread [ns]
        num_paths             int   valid path count
        los_exists            bool  True if at least one LoS path found
        sat_x_m               float proxy TX x-position [m]
        sat_y_m               float proxy TX y-position [m]
        sampled_ues           int   number of UE sample points aggregated
"""

import math
import numpy as np

# Mitsuba's cuda_ad_mono_polarized variant requires OptiX for scene/BVH work.
# Check for libnvoptix before committing to the CUDA variant; fall back to the
# LLVM JIT variant (CPU) when OptiX is absent (e.g. Jetson Orin without OptiX).
import ctypes.util as _ctypes_util
import mitsuba as mi
if mi.variant() is None:
    if _ctypes_util.find_library("nvoptix"):
        mi.set_variant("cuda_ad_mono_polarized")
    else:
        mi.set_variant("llvm_ad_mono_polarized")

import sionna
import sionna.rt
from sionna.rt import (
    load_scene,
    PlanarArray,
    Transmitter,
    Receiver,
    Camera,
    PathSolver,
    RadioMapSolver,
)

from config import (
    RT_SCENE_NAME,
    RT_SCENE_FREQ_HZ,
    RT_MAX_DEPTH,
    RT_SAMPLES_PER_TX,
    RT_CELL_SIZE,
    RT_UE_POSITION,
    RT_UE_SAMPLE_POSITIONS,
    RT_SAT_SCENE_HEIGHT_M,
    RT_SAT_INITIAL_ZENITH_DEG,
    RT_CAM_POSITION,
    RT_CAM_LOOK_AT,
    RT_RENDER_PATHS,
    RT_RENDER_NUM_SAMPLES,
    NUM_SATELLITES,
    SAT_SPACING_DEG,
    SAT_HANDOVER_ELEVATION_DEG,
    SAT_HEIGHT_M,
    RT_TX_POWER_DBM,
)


# =============================================================================
# Satellite geometry helpers
# =============================================================================

def _sat_elevation_deg(sat_x: float, sat_y: float, sat_z: float,
                       ue_x: float, ue_y: float, ue_z: float) -> float:
    """
    Compute the elevation angle [degrees] from the UE to the satellite
    proxy transmitter using simple Euclidean geometry within the scene.

    Parameters
    ----------
    sat_x, sat_y, sat_z : float  Satellite proxy position [m].
    ue_x, ue_y, ue_z    : float  UE position [m].

    Returns
    -------
    float  Elevation angle in degrees (0 = horizon, 90 = zenith).
    """
    dx = sat_x - ue_x
    dy = sat_y - ue_y
    dz = sat_z - ue_z
    return math.degrees(math.atan2(dz, math.hypot(dx, dy)))


def _propagation_delay_ms(height_m: float, elev_deg: float) -> float:
    """
    One-way propagation delay [ms] from ground to satellite at the real
    orbital altitude, using spherical-Earth slant-range geometry.

    Parameters
    ----------
    height_m : float  Satellite orbital altitude [m] (SAT_HEIGHT_M = 550 km).
    elev_deg : float  Elevation angle from the ground receiver [degrees].
    """
    RE = 6_371_000.0
    e  = math.radians(max(elev_deg, 0.1))   # guard against zero/negative
    slant = (math.sqrt((RE + height_m) ** 2 - (RE * math.cos(e)) ** 2)
             - RE * math.sin(e))
    return slant / 3e8 * 1e3


def _satellite_positions(ue_pos: list, height_m: float,
                         n_sats: int, spacing_deg: float,
                         initial_zenith_deg: float = 0.0) -> list:
    """
    Generate proxy transmitter positions for N satellites distributed at
    equal angular spacing around the UE in the horizontal plane, each at
    height ``height_m``.

    The first satellite is placed at ``initial_zenith_deg`` from zenith;
    subsequent ones are offset by ``spacing_deg`` so they represent the
    constellation passing from high elevation toward the horizon.

    Parameters
    ----------
    ue_pos    : [x, y, z]  UE position in the scene [m].
    height_m  : float      Proxy altitude above ground [m].
    n_sats    : int        Number of satellite positions to generate.
    spacing_deg : float    Angular spacing between satellites [degrees].
                           This is converted to a horizontal offset using
                           tan(zenith_angle) × height so that the elevation
                           angle decreases by spacing_deg per satellite step.
    initial_zenith_deg : float
                           Zenith angle of the first (highest-elevation)
                           satellite [degrees].  Setting this > 0 avoids
                           placing a transmitter directly overhead, which
                           returns 0 valid paths in scenes where no vertical
                           surfaces are illuminated at near-zero incidence.

    Returns
    -------
    list of (x, y, z) tuples, one per satellite proxy.
    """
    positions = []
    ux, uy, _ = ue_pos

    for i in range(n_sats):
        # Zenith angle increases with index; first satellite starts at
        # initial_zenith_deg so no satellite is placed directly overhead.
        zenith_deg = initial_zenith_deg + i * spacing_deg
        zenith_rad = math.radians(zenith_deg)

        # Horizontal distance from UE so that tan(zenith) = horiz / height
        horiz = height_m * math.tan(zenith_rad)

        # Spread satellites along the X-axis for a simple planar pass.
        sat_x = ux + horiz
        sat_y = uy
        sat_z = height_m

        positions.append((sat_x, sat_y, sat_z))

    return positions


# =============================================================================
# Channel statistics extraction
# =============================================================================

def _proxy_fspl_db(sat_pos: tuple, ue_pos: list) -> float:
    """
    Free-space path loss [dB] from the proxy satellite TX to a UE position,
    at the scene carrier frequency.

    This is used to normalise RT path gains by removing the proxy-distance
    contribution so that the remaining 'urban correction' reflects only the
    building-shadow / multipath effect — not an artefact of how far the proxy
    TX happens to sit from the UE in the scene.

    For example, a proxy at 1776 m from the UE naturally shows ~12 dB lower
    raw RT gain than a proxy at 506 m at the same building-shadow level; the
    normalisation removes this geometric bias before computing inter-satellite
    urban corrections.
    """
    dx = sat_pos[0] - ue_pos[0]
    dy = sat_pos[1] - ue_pos[1]
    dz = sat_pos[2] - ue_pos[2]
    d  = math.sqrt(dx * dx + dy * dy + dz * dz)
    d  = max(d, 1.0)   # guard: avoid log(0) for degenerate positions
    lam = 3e8 / RT_SCENE_FREQ_HZ
    return 20.0 * math.log10(4.0 * math.pi * d / lam)


def _extract_channel_stats(paths, sat_id: int,
                             sat_pos: tuple, elev_deg: float,
                             rx_idx: int = 0,
                             ue_pos: list = None) -> dict:
    """
    Compute scalar channel statistics from a Sionna RT Paths object.

    The statistics summarise the urban multipath environment seen by the
    UE when looking towards this satellite direction.  They are used in
    ntn_ns3.py to calibrate the packet error rate model.

    Parameters
    ----------
    paths    : sionna.rt.Paths  Computed propagation paths.
    sat_id   : int              Satellite index (0-based).
    sat_pos  : (x, y, z)       Satellite proxy position [m].
    elev_deg : float            Elevation angle [deg].
    rx_idx   : int              Receiver index to extract (default 0).
                                With a single-scene multi-receiver setup the
                                Paths tensors have shape
                                [num_rx, num_rx_ant, num_tx, num_tx_ant, max_paths].
                                Slicing by rx_idx isolates one UE's paths.
    ue_pos   : [x, y, z]       UE position in the scene [m].  Used to compute
                                the proxy FSPL for gain normalisation.

    Returns
    -------
    dict  Channel statistics (see module docstring for keys).
    """
    if ue_pos is None:
        ue_pos = list(RT_UE_POSITION)

    # Proxy FSPL: free-space loss from the proxy TX to this UE position [dB].
    # Subtracting this from the raw RT gain yields a 'normalised gain' that
    # represents only the shadow-fading / multipath correction relative to
    # free space, independent of how far the proxy sits from the UE.
    pfspl = _proxy_fspl_db(sat_pos, ue_pos)

    # paths.a returns (real_tensor, imag_tensor), each shaped
    # [num_rx, num_rx_ant, num_tx, num_tx_ant, max_paths].
    # Slice receiver rx_idx and flatten the remaining antenna/tx dims.
    tau_np   = np.array(paths.tau)[rx_idx].flatten()    # [max_paths]
    a_re_np  = np.array(paths.a[0])[rx_idx].flatten()   # [max_paths]
    a_im_np  = np.array(paths.a[1])[rx_idx].flatten()   # [max_paths]
    valid_np = np.array(paths.valid)[rx_idx].flatten().astype(bool)  # [max_paths]

    a_mag = np.hypot(a_re_np, a_im_np)  # |h|, linear

    valid_tau = tau_np[valid_np]
    valid_a   = a_mag[valid_np]

    num_paths = int(valid_np.sum())

    if num_paths == 0:
        # No paths resolved (e.g. satellite below horizon / blocked)
        return dict(
            sat_id               = sat_id,
            elevation_deg        = round(elev_deg, 1),
            mean_path_gain_db    = -200.0,   # Effectively no signal
            normalized_gain_db   = float("nan"),
            proxy_fspl_db        = round(pfspl, 2),
            delay_spread_ns      = 0.0,
            num_paths            = 0,
            los_exists           = False,
            sat_x_m              = sat_pos[0],
            sat_y_m              = sat_pos[1],
        )

    # Mean channel gain: average |h|² (power), then convert to dB.
    # Correct per Jensen's inequality — averaging power then taking
    # 10·log10 is unbiased; averaging amplitude then 20·log10 overestimates.
    mean_power_lin = float(np.mean(valid_a ** 2))
    mean_gain_db   = float(10.0 * np.log10(mean_power_lin + 1e-60))

    # Normalised gain: raw RT gain minus proxy FSPL.
    # This represents shadow fading relative to free space at the proxy
    # distance, removing the geometric bias introduced by different proxy
    # positions.  Used by ns3.py instead of mean_path_gain_db when computing
    # inter-satellite urban corrections.
    normalized_gain_db = round(mean_gain_db + pfspl, 2)

    # Power-weighted RMS delay spread [ns].
    # Standard definition: σ_τ = sqrt(Σ p_k(τ_k − τ̄)² / Σ p_k)
    # where p_k = |h_k|² (path power weight).
    # Equal-weight std overestimates spread by giving weak late paths
    # the same importance as the strong early paths.
    power_weights   = valid_a ** 2 / (valid_a ** 2).sum()
    mean_tau        = float((power_weights * valid_tau).sum())
    delay_spread_ns = float(
        np.sqrt((power_weights * (valid_tau - mean_tau) ** 2).sum()) * 1e9
    )

    # LoS test by geometry: the earliest RT arrival must match the
    # free-space delay between UE and proxy TX (within ~5 ns tolerance for
    # numeric / propagation-model jitter).  This is a stricter, physically
    # grounded criterion than the old amplitude heuristic — a weak diffracted
    # ray arriving first would otherwise be mis-classified as LoS and give
    # spurious K-factor values.
    earliest_idx = int(np.argmin(valid_tau))
    geom_dist_m  = float(np.linalg.norm(
        np.asarray(sat_pos, dtype=float) - np.asarray(ue_pos, dtype=float)))
    geom_tau_s   = geom_dist_m / 3e8
    los_exists   = bool(abs(float(valid_tau[earliest_idx]) - geom_tau_s) < 5e-9)

    # Rician K-factor (3GPP definition): ratio of dominant (LoS) path power to
    # the sum of diffuse (non-dominant) path powers.  Using the strongest path
    # — not the earliest — makes K robust to weak first-arrival rays and aligns
    # with the TR 38.811 "K = P_specular / P_diffuse" convention.  Only defined
    # when a geometric LoS exists; NaN otherwise (NLoS → Rayleigh, no K).
    dominant_idx      = int(np.argmax(valid_a))
    dom_power_lin     = float(valid_a[dominant_idx] ** 2)
    total_power_lin   = float(np.sum(valid_a ** 2))
    scatter_power_lin = total_power_lin - dom_power_lin
    if los_exists and scatter_power_lin > 1e-60:
        k_factor_db = round(float(10.0 * np.log10(dom_power_lin / scatter_power_lin)), 2)
    else:
        k_factor_db = float("nan")

    return dict(
        sat_id             = sat_id,
        elevation_deg      = round(elev_deg, 1),
        mean_path_gain_db  = round(mean_gain_db, 2),
        normalized_gain_db = normalized_gain_db,
        proxy_fspl_db      = round(pfspl, 2),
        delay_spread_ns    = round(delay_spread_ns, 2),
        num_paths          = num_paths,
        los_exists         = los_exists,
        k_factor_db        = k_factor_db,
        sat_x_m            = round(sat_pos[0], 1),
        sat_y_m            = round(sat_pos[1], 1),
    )


def _aggregate_sample_stats(sample_stats: list, sat_id: int, sat_pos: tuple) -> dict:
    """
    Aggregate per-UE-sample channel stats into one per-satellite record.

    Aggregation uses mean values for gain and delay spread, plus a conservative
    10th-percentile gain which is consumed by NS-3 as an urban diversity penalty.
    """
    valid = [s for s in sample_stats if s.get("num_paths", 0) > 0]
    if not valid:
        return dict(
            sat_id=sat_id,
            elevation_deg=0.0,
            mean_path_gain_db=-200.0,
            mean_path_gain_p10_db=-200.0,
            normalized_gain_db=float("nan"),
            normalized_p10_db=float("nan"),
            delay_spread_ns=0.0,
            num_paths=0,
            los_exists=False,
            sat_x_m=round(sat_pos[0], 1),
            sat_y_m=round(sat_pos[1], 1),
            sampled_ues=len(sample_stats),
        )

    gains = np.array([s["mean_path_gain_db"] for s in valid], dtype=float)
    dss   = np.array([s["delay_spread_ns"]   for s in valid], dtype=float)
    elevs = np.array([s["elevation_deg"]     for s in sample_stats], dtype=float)

    # Delay spread is log-normal distributed (3GPP TR 38.811): aggregate with
    # geometric mean so that a single sample with a large canyon-resonant DS
    # doesn't dominate the per-satellite summary.
    dss_positive = dss[dss > 0]
    if len(dss_positive) > 0:
        ds_geo_mean_ns = float(np.exp(np.mean(np.log(dss_positive))))
    else:
        ds_geo_mean_ns = 0.0

    # Normalised gains: raw RT gain + proxy FSPL → shadow fading only.
    # Filter out NaN (from samples with 0 paths) before aggregating.
    norm_gains = np.array([s.get("normalized_gain_db", float("nan")) for s in valid], dtype=float)
    norm_valid = norm_gains[~np.isnan(norm_gains)]
    if len(norm_valid) > 0:
        agg_normalized_gain_db = round(float(np.mean(norm_valid)), 2)
        agg_normalized_p10_db  = round(float(np.percentile(norm_valid, 10)), 2)
    else:
        agg_normalized_gain_db = float("nan")
        agg_normalized_p10_db  = float("nan")

    # Aggregate K-factor: mean over UE samples that have a valid (non-NaN) value.
    k_vals = np.array([s.get("k_factor_db", float("nan")) for s in valid], dtype=float)
    k_valid = k_vals[~np.isnan(k_vals)]
    k_factor_db_agg = round(float(np.mean(k_valid)), 2) if len(k_valid) > 0 else float("nan")

    return dict(
        sat_id                = sat_id,
        elevation_deg         = round(float(np.mean(elevs)), 1),
        mean_path_gain_db     = round(float(np.mean(gains)), 2),
        mean_path_gain_p10_db = round(float(np.percentile(gains, 10)), 2),
        normalized_gain_db    = agg_normalized_gain_db,
        normalized_p10_db     = agg_normalized_p10_db,
        delay_spread_ns       = round(ds_geo_mean_ns, 2),
        num_paths             = int(round(float(np.mean([s["num_paths"] for s in valid])))),
        los_exists            = bool(any(s.get("los_exists", False) for s in sample_stats)),
        k_factor_db           = k_factor_db_agg,
        sat_x_m               = round(sat_pos[0], 1),
        sat_y_m               = round(sat_pos[1], 1),
        sampled_ues           = len(sample_stats),
    )


def _load_scene_with_ues(ue_positions: list) -> tuple:
    """
    Load the Munich scene once and add ALL UE sample positions as receivers.

    Using a single scene with N receivers is significantly faster than loading
    N separate scenes because PathSolver traces rays for all receivers in a
    single GPU/CPU kernel launch, sharing geometry intersection work.

    Returns
    -------
    scene     : sionna.rt.Scene
    receivers : list[sionna.rt.Receiver]  (one per UE position, in order)
    """
    # Resolve RT_SCENE_NAME → either a built-in Sionna scene attribute or a
    # custom Mitsuba XML path.  This lets users swap urban environments by
    # changing one config line (e.g. "munich" → "etoile" or a custom scene).
    scene_ref = getattr(sionna.rt.scene, RT_SCENE_NAME, RT_SCENE_NAME)
    scene = load_scene(scene_ref, merge_shapes=True)
    scene.frequency = RT_SCENE_FREQ_HZ

    # Satellite TX array: single-element with TR38.901 pattern,
    # vertical polarisation (representative of a patch antenna on a
    # satellite that radiates towards the ground).
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="V",
    )

    # RX array: single dipole with cross-polarisation (used for the UE
    # receiver — realistic for a hand-held phone with arbitrary
    # polarisation alignment).
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="cross",
    )

    receivers = []
    for idx, ue_pos in enumerate(ue_positions):
        rx = Receiver(name=f"ue{idx}", position=ue_pos, display_radius=3)
        scene.add(rx)
        receivers.append(rx)

    return scene, receivers


def _make_camera() -> Camera:
    """Overhead camera for scene renders."""
    return Camera(position=RT_CAM_POSITION, look_at=RT_CAM_LOOK_AT)


# =============================================================================
# Per-satellite ray tracing
# =============================================================================

def _trace_satellite(scene, sat_id: int, sat_pos: tuple,
                     ue_positions: list,
                     render_paths: bool = False) -> tuple:
    """
    Add the satellite proxy transmitter to the shared scene, run ONE
    PathSolver call that covers all UE receivers simultaneously, then
    remove the TX.

    Using a shared scene with multiple receivers is the key performance
    optimisation: instead of N_ue separate PathSolver calls (one scene each),
    we make a single call whose GPU/LLVM kernel traverses the BVH once for
    all (TX, RX) pairs.  With N_ue=4 this reduces PathSolver calls from
    4 × num_sats to 1 × num_sats — a 4× speedup for the RT stage.

    Parameters
    ----------
    scene        : sionna.rt.Scene  Contains all UE receivers (ue0…ueN).
    sat_id       : int
    sat_pos      : (x, y, z)  Proxy transmitter position [m].
    ue_positions : list of [x,y,z]  UE positions in scene order (ue0, ue1, …).

    Returns
    -------
    paths       : sionna.rt.Paths
    sample_stats: list[dict]  Per-UE-receiver channel statistics.
    """
    tx_name = f"sat{sat_id}"
    # Steer satellite beam towards the primary (first) UE position.
    # With all UEs within ~200 m of each other and the proxy TX at 300 m,
    # the boresight difference is < 5° for all UE positions — negligible.
    tx = Transmitter(
        name      = tx_name,
        position  = sat_pos,
        look_at   = ue_positions[0],
        velocity  = (0.0, 0.0, 0.0),
        power_dbm = RT_TX_POWER_DBM,
        display_radius = 5,
    )
    scene.add(tx)

    # Single PathSolver call — paths for all UE receivers at once.
    solver = PathSolver()
    paths  = solver(
        scene               = scene,
        max_depth           = RT_MAX_DEPTH,
        los                 = True,
        specular_reflection = True,
        diffuse_reflection  = False,
        refraction          = True,
        synthetic_array     = False,
        seed                = 42 + sat_id,
    )

    # Render paths image using receiver 0 (primary UE).
    # Controlled by RT_RENDER_PATHS; set False in config to skip and save time.
    if render_paths and RT_RENDER_PATHS:
        cam      = _make_camera()
        out_file = f"output/ntn_rt_paths_sat{sat_id}.png"
        try:
            scene.render_to_file(
                camera      = cam,
                filename    = out_file,
                paths       = paths,
                clip_at     = RT_SAT_SCENE_HEIGHT_M + 50,
                resolution  = (1280, 720),
                num_samples = RT_RENDER_NUM_SAMPLES,
            )
            print(f"    [Render] {out_file}")
        except Exception as e:
            print(f"    [Render] Skipped {out_file}: {e}")

    # Extract per-UE stats by slicing the rx_idx dimension of the Paths tensors.
    # Pass ue_pos so that _extract_channel_stats can normalise the raw RT gain
    # by the proxy FSPL from that specific UE location.
    sample_stats = []
    for rx_idx, ue_pos in enumerate(ue_positions):
        elev_i = _sat_elevation_deg(*sat_pos, *ue_pos)
        st = _extract_channel_stats(paths, sat_id, sat_pos, elev_i,
                                    rx_idx=rx_idx, ue_pos=list(ue_pos))
        sample_stats.append(st)

    scene.remove(tx_name)
    return paths, sample_stats


# =============================================================================
# Radio map
# =============================================================================

def _compute_and_save_radiomap(scene) -> None:
    """
    Compute a 2-D path-gain radio map with all satellite transmitters
    added simultaneously, then save the render.  This gives a composite
    view of coverage from the whole constellation snapshot.
    """
    print("\n  Computing composite radio map ...")
    solver = RadioMapSolver()
    try:
        rm = solver(
            scene         = scene,
            max_depth     = RT_MAX_DEPTH,
            cell_size     = RT_CELL_SIZE,
            samples_per_tx = RT_SAMPLES_PER_TX,
        )
        cam = _make_camera()
        scene.render_to_file(
            camera   = cam,
            filename = "output/ntn_rt_radiomap.png",
            radio_map = rm,
            rm_metric  = "path_gain",
            rm_db_scale = True,
            resolution  = (1280, 720),
            num_samples = RT_RENDER_NUM_SAMPLES,
        )
        print("  [Render] output/ntn_rt_radiomap.png")
    except Exception as e:
        print(f"  [RadioMap] Skipped: {e}")


# =============================================================================
# Public entry point
# =============================================================================

def run_ray_tracing() -> list:
    """
    Execute the full Sionna RT ray tracing pipeline:

    1. Load the Munich scene and place the UE receiver.
    2. For each satellite in the constellation snapshot:
       a. Compute its proxy position and elevation angle to the UE.
       b. Run PathSolver (LoS + specular reflections + refractions).
       c. Extract service-link channel statistics across sampled UEs.
       d. Save a scene render with paths overlaid.
    3. Compute and save a composite radio map.

    Parameters
    ----------
    None — all settings come from config.py.

    Returns
    -------
    channel_stats : list[dict]
        One dict per satellite (in order of simulation index).
         Keys: sat_id, elevation_deg, mean_path_gain_db,
               mean_path_gain_p10_db, delay_spread_ns, num_paths,
               los_exists, sat_x_m, sat_y_m, sampled_ues.
        This list is passed directly to ntn_ns3.run_ns3() so that the
        NS-3 link budget uses RT-informed channel parameters.
    """
    print(f"\n[Sionna RT]  5G-NTN '{RT_SCENE_NAME}' scene — satellite constellation pass")
    print(f"  Mitsuba variant: {mi.variant()}")
    print(f"  Frequency      : {RT_SCENE_FREQ_HZ/1e9:.2f} GHz")
    print(f"  UE samples     : {len(RT_UE_SAMPLE_POSITIONS)} points (multi-RX single scene)")
    print(f"  Primary UE     : {RT_UE_POSITION} m")
    print(f"  Proxy altitude : {RT_SAT_SCENE_HEIGHT_M} m")
    print(f"  Satellites     : {NUM_SATELLITES}  "
          f"(initial zenith {RT_SAT_INITIAL_ZENITH_DEG}°, "
          f"spacing {SAT_SPACING_DEG}° per step)\n")

    # ── Build satellite proxy positions ───────────────────────────────────────
    sat_positions = _satellite_positions(
        ue_pos             = RT_UE_POSITION,
        height_m           = RT_SAT_SCENE_HEIGHT_M,
        n_sats             = NUM_SATELLITES,
        spacing_deg        = SAT_SPACING_DEG,
        initial_zenith_deg = RT_SAT_INITIAL_ZENITH_DEG,
    )

    # ── Load ONE scene with all UE receivers ──────────────────────────────────
    # Previously the code loaded N separate scenes (one per UE) and called
    # PathSolver N times per satellite.  Loading a single shared scene and
    # adding all UE positions as named receivers (ue0…ueN) lets PathSolver
    # trace all (TX, RX) pairs in a single GPU kernel launch, cutting the
    # number of solver calls from N_ue × N_sats → 1 × N_sats.
    ue_samples = RT_UE_SAMPLE_POSITIONS if RT_UE_SAMPLE_POSITIONS else [RT_UE_POSITION]
    scene, _receivers = _load_scene_with_ues(ue_samples)

    all_stats = []

    for sat_id, sat_pos in enumerate(sat_positions):
        elev_primary = _sat_elevation_deg(*sat_pos, *RT_UE_POSITION)
        visible = elev_primary >= SAT_HANDOVER_ELEVATION_DEG
        print(f"  Satellite {sat_id}:  UE elev={elev_primary:5.1f}°  "
              f"pos=({sat_pos[0]:6.1f}, {sat_pos[1]:5.1f}, {sat_pos[2]:.0f}) m  "
              f"{'[visible]' if visible else '[below horizon threshold]'}")

        # One PathSolver call → sample_stats list (one dict per UE receiver)
        _, sample_stats = _trace_satellite(
            scene,
            sat_id,
            sat_pos,
            ue_samples,
            render_paths=True,   # always render from primary UE (rx0)
        )

        stats = _aggregate_sample_stats(sample_stats, sat_id, sat_pos)
        all_stats.append(stats)

        print(f"    svc:    paths={stats['num_paths']}  "
              f"gain={stats['mean_path_gain_db']:.1f} dB  "
              f"p10={stats.get('mean_path_gain_p10_db', stats['mean_path_gain_db']):.1f} dB  "
              f"ds={stats['delay_spread_ns']:.1f} ns")

    # ── Composite radio map: add all visible sats back simultaneously ─────────
    visible_sats = [
        (s["sat_id"], sat_positions[s["sat_id"]])
        for s in all_stats
        if s["elevation_deg"] >= SAT_HANDOVER_ELEVATION_DEG
    ]
    for sat_id, sat_pos in visible_sats:
        tx = Transmitter(
            name      = f"sat{sat_id}",
            position  = sat_pos,
            look_at   = RT_UE_POSITION,
            power_dbm = RT_TX_POWER_DBM,
        )
        scene.add(tx)

    _compute_and_save_radiomap(scene)

    print(f"\n  Ray tracing complete.  {len(all_stats)} satellite snapshots computed.\n")
    return all_stats
