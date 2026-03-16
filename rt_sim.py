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

Two receivers are placed in the scene:
  - UE  at RT_UE_POSITION  — service link (sat → UE)
  - GS  at RT_GS_POSITION  — feeder link  (sat → Ground Station)

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
        mean_path_gain_db     float mean |h| over valid paths [dB]
        delay_spread_ns       float RMS delay spread [ns]
        num_paths             int   valid path count
        los_exists            bool  True if at least one LoS path found
        sat_x_m               float proxy TX x-position [m]
        sat_y_m               float proxy TX y-position [m]
      Feeder-link keys (sat → GS):
        feeder_elevation_deg      float GS elevation angle [deg]
        feeder_mean_path_gain_db  float mean |h| over valid feeder paths [dB]
        feeder_delay_spread_ns    float RMS feeder delay spread [ns]
        feeder_num_paths          int   valid feeder path count
        feeder_propagation_delay_ms float one-way propagation delay [ms]
                                        computed from GS elevation geometry
"""

import math
import numpy as np

# Force LLVM (CPU) Mitsuba variant BEFORE importing sionna.rt.
# Without this, Sionna tries cuda_ad_mono_polarized first; on machines
# without CUDA drivers the DrJIT GPU init crashes with a fatal GIL error.
import mitsuba as mi
if mi.variant() is None:
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
    RT_SCENE_FREQ_HZ,
    RT_MAX_DEPTH,
    RT_SAMPLES_PER_TX,
    RT_CELL_SIZE,
    RT_UE_POSITION,
    RT_GS_POSITION,
    RT_SAT_SCENE_HEIGHT_M,
    RT_SAT_INITIAL_ZENITH_DEG,
    RT_CAM_POSITION,
    RT_CAM_LOOK_AT,
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
    horiz = math.sqrt(dx ** 2 + dy ** 2)
    return math.degrees(math.atan2(dz, horiz))


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

def _extract_channel_stats(paths, sat_id: int,
                            sat_pos: tuple, elev_deg: float) -> dict:
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

    Returns
    -------
    dict  Channel statistics (see module docstring for keys).
    """
    tau_np  = np.array(paths.tau)            # [num_rx, …, num_paths]
    a_re_np = np.array(paths.a[0])           # real part of channel coefficients
    a_im_np = np.array(paths.a[1])           # imaginary part
    valid_np = np.array(paths.valid).astype(bool)

    a_mag = np.sqrt(a_re_np ** 2 + a_im_np ** 2)  # |h|, linear

    valid_flat = valid_np.flatten()
    tau_flat   = tau_np.flatten()
    a_flat     = a_mag.flatten()

    valid_tau = tau_flat[valid_flat]
    valid_a   = a_flat[valid_flat]

    num_paths = int(valid_flat.sum())

    if num_paths == 0:
        # No paths resolved (e.g. satellite below horizon / blocked)
        return dict(
            sat_id          = sat_id,
            elevation_deg   = round(elev_deg, 1),
            mean_path_gain_db = -200.0,   # Effectively no signal
            delay_spread_ns = 0.0,
            num_paths       = 0,
            los_exists      = False,
            sat_x_m         = sat_pos[0],
            sat_y_m         = sat_pos[1],
        )

    # Mean channel gain: average |h|² (power), then convert to dB.
    # Correct per Jensen's inequality — averaging power then taking
    # 10·log10 is unbiased; averaging amplitude then 20·log10 overestimates.
    mean_power_lin = float(np.mean(valid_a ** 2))
    mean_gain_db   = float(10.0 * np.log10(mean_power_lin + 1e-60))

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

    # LoS exists if the earliest-arriving (minimum-delay) path has a gain
    # within 6 dB of the strongest path.  This heuristic is necessary because
    # Sionna RT does not expose path-type flags (LoS vs NLoS) on the Paths
    # object in this version.  6 dB ↔ amplitude ratio of 0.5 (power ratio 0.25).
    earliest_idx = int(np.argmin(valid_tau))
    los_exists   = bool(valid_a[earliest_idx] >= 0.5 * valid_a.max())

    return dict(
        sat_id          = sat_id,
        elevation_deg   = round(elev_deg, 1),
        mean_path_gain_db = round(mean_gain_db, 2),
        delay_spread_ns = round(delay_spread_ns, 2),
        num_paths       = num_paths,
        los_exists      = los_exists,
        sat_x_m         = round(sat_pos[0], 1),
        sat_y_m         = round(sat_pos[1], 1),
    )


def _extract_feeder_stats(paths, sat_id: int,
                           sat_pos: tuple, gs_elev_deg: float) -> dict:
    """
    Compute feeder-link channel statistics from a Sionna RT Paths object,
    using the Ground Station (GS) receiver slice of the paths tensor.

    Sionna RT stores paths for all receivers in a single tensor with shape
    [..., num_rx, ...].  Receiver index 0 = UE, receiver index 1 = GS.
    This function extracts the GS slice (index 1) to characterise the
    satellite → GS feeder link.

    The propagation delay is computed analytically from the GS elevation
    angle and the real 550 km orbital altitude (SAT_HEIGHT_M), because the
    proxy TX is only 300 m above the scene and the RT path delays do not
    represent the true 550 km slant range.

    Parameters
    ----------
    paths       : sionna.rt.Paths  Computed propagation paths (both receivers).
    sat_id      : int              Satellite index (0-based).
    sat_pos     : (x, y, z)        Satellite proxy position [m].
    gs_elev_deg : float            Elevation angle from GS to satellite [deg].

    Returns
    -------
    dict with keys:
        feeder_elevation_deg      float
        feeder_mean_path_gain_db  float  [dB]; −200 if no paths
        feeder_delay_spread_ns    float  [ns]
        feeder_num_paths          int
        feeder_propagation_delay_ms float  analytic one-way delay [ms]
    """
    # Receiver index 1 = GS (0 = UE).  Sionna RT path tensors have shape
    # [2, num_rx, num_tx, rx_ant, tx_ant, max_paths] where axis-0 is
    # (real, imaginary).  We select receiver 1 along axis 1.
    num_paths = 0
    valid_a   = np.array([], dtype=np.float64)
    valid_tau = np.array([], dtype=np.float64)
    try:
        tau_np   = np.array(paths.tau)       # [..., num_rx, ..., num_paths]
        a_re_np  = np.array(paths.a[0])      # real part
        a_im_np  = np.array(paths.a[1])      # imaginary part
        valid_np = np.array(paths.valid).astype(bool)

        # Select GS receiver (index 1) along the num_rx axis.
        # paths.tau shape: (num_rx, num_tx, rx_ant, tx_ant, max_paths)
        # paths.a   shape: (num_rx, num_tx, rx_ant, tx_ant, max_paths)
        # paths.valid shape: (num_rx, num_tx, rx_ant, tx_ant, max_paths)
        # (Sionna 1.x — receiver axis is axis 0 of the non-real/imag dims)
        if tau_np.ndim >= 1 and tau_np.shape[0] >= 2:
            tau_gs   = tau_np[1]
            a_re_gs  = a_re_np[1]
            a_im_gs  = a_im_np[1]
            valid_gs = valid_np[1]
        else:
            # Fallback: single receiver — feeder channel not available
            raise ValueError("paths tensor has fewer than 2 receivers")

        a_mag_gs   = np.sqrt(a_re_gs ** 2 + a_im_gs ** 2)
        valid_flat = valid_gs.flatten()
        tau_flat   = tau_gs.flatten()
        a_flat     = a_mag_gs.flatten()

        valid_tau = tau_flat[valid_flat]
        valid_a   = a_flat[valid_flat]
        num_paths = int(valid_flat.sum())

    except Exception as exc:
        # If the tensor structure differs from expectation, fall back gracefully
        print(f"    [Feeder] Could not extract GS paths for sat{sat_id}: {exc}")
        num_paths = 0

    prop_delay_ms = _propagation_delay_ms(SAT_HEIGHT_M, gs_elev_deg)

    if num_paths == 0:
        return dict(
            feeder_elevation_deg      = round(gs_elev_deg, 1),
            feeder_mean_path_gain_db  = -200.0,
            feeder_delay_spread_ns    = 0.0,
            feeder_num_paths          = 0,
            feeder_propagation_delay_ms = round(prop_delay_ms, 3),
        )

    mean_power_lin  = float(np.mean(valid_a ** 2))
    mean_gain_db    = float(10.0 * np.log10(mean_power_lin + 1e-60))
    power_weights   = valid_a ** 2 / (valid_a ** 2).sum()
    mean_tau_f      = float((power_weights * valid_tau).sum())
    delay_spread_ns = float(
        np.sqrt((power_weights * (valid_tau - mean_tau_f) ** 2).sum()) * 1e9
    )

    return dict(
        feeder_elevation_deg      = round(gs_elev_deg, 1),
        feeder_mean_path_gain_db  = round(mean_gain_db, 2),
        feeder_delay_spread_ns    = round(delay_spread_ns, 2),
        feeder_num_paths          = num_paths,
        feeder_propagation_delay_ms = round(prop_delay_ms, 3),
    )

def _load_scene_with_ue(ue_pos: list, gs_pos: list) -> tuple:
    """
    Load the Munich scene, set the carrier frequency, and place both the UE
    and the Ground Station receivers in the scene.

    Two receivers allow a single PathSolver call to return paths toward both
    the UE (service link, receiver index 0) and the GS (feeder link,
    receiver index 1) simultaneously.

    Returns
    -------
    scene  : sionna.rt.Scene
    rx_ue  : sionna.rt.Receiver  (the UE, receiver index 0)
    rx_gs  : sionna.rt.Receiver  (the Ground Station, receiver index 1)
    """
    scene = load_scene(sionna.rt.scene.munich, merge_shapes=True)
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

    # RX array: single dipole with cross-polarisation (used for both UE
    # and GS receivers — realistic for a hand-held phone and a gateway
    # dish that may have arbitrary polarisation alignment).
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="cross",
    )

    # UE (phone) — service link receiver (index 0)
    rx_ue = Receiver(name="ue", position=ue_pos, display_radius=3)
    scene.add(rx_ue)

    # Ground Station — feeder link receiver (index 1)
    rx_gs = Receiver(name="gs", position=gs_pos, display_radius=5)
    scene.add(rx_gs)

    return scene, rx_ue, rx_gs


def _make_camera() -> Camera:
    """Overhead camera for scene renders."""
    return Camera(position=RT_CAM_POSITION, look_at=RT_CAM_LOOK_AT)


# =============================================================================
# Per-satellite ray tracing
# =============================================================================

def _trace_satellite(scene, sat_id: int, sat_pos: tuple,
                     elev_deg: float) -> tuple:
    """
    Add the satellite proxy transmitter to the scene, run PathSolver for
    both receivers (UE and GS), render a paths image, then remove the TX.

    Parameters
    ----------
    scene    : sionna.rt.Scene  (must already contain "ue" and "gs" receivers)
    sat_id   : int
    sat_pos  : (x, y, z)  Proxy transmitter position [m].
    elev_deg : float       UE elevation angle [deg] for logging.

    Returns
    -------
    paths : sionna.rt.Paths
    stats : dict  Service-link + feeder-link channel statistics merged.
    """
    tx_name = f"sat{sat_id}"
    tx = Transmitter(
        name     = tx_name,
        position = sat_pos,
        look_at  = RT_UE_POSITION,   # Beam steered towards the UE
        velocity = (0.0, 0.0, 0.0),  # Static within this snapshot
        power_dbm = RT_TX_POWER_DBM, # from config.RT_TX_POWER_DBM
        display_radius = 5,
    )
    scene.add(tx)

    # Path computation — both UE and GS receivers are in the scene;
    # Sionna RT returns paths to all receivers in a single call.
    solver = PathSolver()
    paths  = solver(
        scene              = scene,
        max_depth          = RT_MAX_DEPTH,
        los                = True,
        specular_reflection = True,
        diffuse_reflection  = False,
        refraction         = True,
        synthetic_array    = False,
        seed               = 42 + sat_id,   # Different seed per satellite
    )

    # Render and save paths image for this satellite
    cam      = _make_camera()
    out_file = f"output/ntn_rt_paths_sat{sat_id}.png"
    try:
        scene.render_to_file(
            camera     = cam,
            filename   = out_file,
            paths      = paths,
            clip_at    = RT_SAT_SCENE_HEIGHT_M + 50,
            resolution = (1280, 720),
            num_samples = 256,
        )
        print(f"    [Render] {out_file}")
    except Exception as e:
        print(f"    [Render] Skipped {out_file}: {e}")

    # Service-link stats (sat → UE)
    stats = _extract_channel_stats(paths, sat_id, sat_pos, elev_deg)

    # Feeder-link stats (sat → GS)
    gs_elev_deg = _sat_elevation_deg(*sat_pos, *RT_GS_POSITION)
    feeder_stats = _extract_feeder_stats(paths, sat_id, sat_pos, gs_elev_deg)
    stats.update(feeder_stats)

    # Remove this satellite before adding the next one
    scene.remove(tx_name)

    return paths, stats


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
            num_samples = 256,
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

    1. Load the Munich scene and place the UE and Ground Station receivers.
    2. For each satellite in the constellation snapshot:
       a. Compute its proxy position and elevation angles to both UE and GS.
       b. Run PathSolver (LoS + specular reflections + refractions) toward
          both receivers simultaneously.
       c. Extract service-link stats (sat→UE) and feeder-link stats (sat→GS).
       d. Save a scene render with paths overlaid.
    3. Compute and save a composite radio map.

    Parameters
    ----------
    None — all settings come from config.py.

    Returns
    -------
    channel_stats : list[dict]
        One dict per satellite (in order of simulation index).
        Service-link keys: sat_id, elevation_deg, mean_path_gain_db,
          delay_spread_ns, num_paths, los_exists, sat_x_m, sat_y_m.
        Feeder-link keys: feeder_elevation_deg, feeder_mean_path_gain_db,
          feeder_delay_spread_ns, feeder_num_paths, feeder_propagation_delay_ms.
        This list is passed directly to ntn_ns3.run_ns3() so that the
        NS-3 link budget uses RT-informed channel parameters for both hops.
    """
    print("\n[Sionna RT]  5G-NTN Munich scene — satellite constellation pass")
    print(f"  Frequency      : {RT_SCENE_FREQ_HZ/1e9:.2f} GHz")
    print(f"  UE position    : {RT_UE_POSITION} m")
    print(f"  GS position    : {RT_GS_POSITION} m")
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

    elev_angles = [
        _sat_elevation_deg(*pos, *RT_UE_POSITION)
        for pos in sat_positions
    ]

    # ── Load scene once, trace each satellite sequentially ───────────────────
    scene, rx_ue, rx_gs = _load_scene_with_ue(RT_UE_POSITION, RT_GS_POSITION)

    all_stats = []

    for sat_id, (sat_pos, elev_deg) in enumerate(zip(sat_positions, elev_angles)):
        visible = elev_deg >= SAT_HANDOVER_ELEVATION_DEG
        gs_elev = _sat_elevation_deg(*sat_pos, *RT_GS_POSITION)
        print(f"  Satellite {sat_id}:  UE elev={elev_deg:5.1f}°  "
              f"GS elev={gs_elev:5.1f}°  "
              f"pos=({sat_pos[0]:6.1f}, {sat_pos[1]:5.1f}, {sat_pos[2]:.0f}) m  "
              f"{'[visible]' if visible else '[below horizon threshold]'}")

        _, stats = _trace_satellite(scene, sat_id, sat_pos, elev_deg)
        all_stats.append(stats)

        print(f"    svc:    paths={stats['num_paths']}  "
              f"gain={stats['mean_path_gain_db']:.1f} dB  "
              f"ds={stats['delay_spread_ns']:.1f} ns")
        print(f"    feeder: paths={stats['feeder_num_paths']}  "
              f"gain={stats['feeder_mean_path_gain_db']:.1f} dB  "
              f"delay={stats['feeder_propagation_delay_ms']:.2f} ms")

    # ── Composite radio map: add all visible sats back simultaneously ─────────
    visible_sats = [
        (i, pos) for i, (pos, e) in enumerate(zip(sat_positions, elev_angles))
        if e >= SAT_HANDOVER_ELEVATION_DEG
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
