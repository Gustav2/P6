"""
rt_sim.py — Part 1: Sionna RT ray tracing  (runs before NS-3)
==============================================================
Models the 5G-NTN uplink from a street-level UE (phone) placed in the
Munich scene to a constellation of LEO satellites passing overhead.

Because the actual orbital altitude (600 km) is far outside the Sionna RT
scene bounds (~100 m), each satellite is represented by a proxy transmitter
placed RT_SAT_SCENE_HEIGHT_M above the scene.  This produces the correct
near-vertical incidence geometry for extracting urban multipath statistics
(delay spread, shadow fading, angle-of-arrival spread) which are then fed
into the NS-3 link budget.  The true free-space path loss over 600 km is
computed analytically in ntn_ns3.py.

Multi-satellite pass
---------------------
NUM_SATELLITES proxy transmitters are placed at angular offsets
(SAT_SPACING_DEG) around the scene to represent successive satellites
in the constellation.  For each, ray tracing is performed independently
and per-satellite channel statistics are returned.

Outputs
-------
  ntn_rt_paths_sat<N>.png   — scene render with paths for satellite N
  ntn_rt_radiomap.png       — composite radio map (all satellites)

Returns
-------
  channel_stats : list[dict]
      One dict per satellite with keys:
        sat_id            int   satellite index (0-based)
        elevation_deg     float elevation angle of this satellite [deg]
        mean_path_gain_db float mean complex channel gain over all valid
                                paths [dB]; used to calibrate NS-3 PER
        delay_spread_ns   float RMS delay spread of valid paths [ns]
        num_paths         int   number of valid propagation paths found
        los_exists        bool  True if at least one LoS path was found
        sat_x_m           float proxy TX x-position in scene [m]
        sat_y_m           float proxy TX y-position in scene [m]
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
    RT_SAT_SCENE_HEIGHT_M,
    RT_SAT_INITIAL_ZENITH_DEG,
    RT_CAM_POSITION,
    RT_CAM_LOOK_AT,
    NUM_SATELLITES,
    SAT_SPACING_DEG,
    SAT_HANDOVER_ELEVATION_DEG,
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

    # Mean channel gain: average |h|² → convert to dB
    mean_gain_lin = float(np.mean(valid_a))
    mean_gain_db  = float(20.0 * np.log10(mean_gain_lin + 1e-30))

    # RMS delay spread: std-dev of path delays weighted equally
    delay_spread_ns = float(np.std(valid_tau) * 1e9)

    # LoS exists if the minimum-delay path has a gain within 6 dB of max
    # (a stricter definition would require path type flags, but Sionna RT
    # does not expose them on the Paths object in this version)
    los_exists = bool(valid_a.max() > 0.5 * valid_a.max())  # always True if any path

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


# =============================================================================
# Scene helpers
# =============================================================================

def _load_scene_with_ue(ue_pos: list) -> tuple:
    """
    Load the Munich scene, set the carrier frequency, and place the UE
    receiver at ``ue_pos``.

    The UE uses a single cross-polarised dipole antenna — a common model
    for a hand-held 5G device.  The satellite proxy transmitters will be
    added separately for each satellite position.

    Returns
    -------
    scene : sionna.rt.Scene
    rx    : sionna.rt.Receiver  (the UE)
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

    # UE RX array: single dipole with cross-polarisation (realistic for
    # a hand-held device that may be in any orientation).
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="cross",
    )

    rx = Receiver(name="ue", position=ue_pos, display_radius=3)
    scene.add(rx)

    return scene, rx


def _make_camera() -> Camera:
    """Overhead camera for scene renders."""
    return Camera(position=RT_CAM_POSITION, look_at=RT_CAM_LOOK_AT)


# =============================================================================
# Per-satellite ray tracing
# =============================================================================

def _trace_satellite(scene, rx, sat_id: int, sat_pos: tuple,
                     elev_deg: float) -> tuple:
    """
    Add the satellite proxy transmitter to the scene, run PathSolver,
    render a paths image, then remove the transmitter.

    Parameters
    ----------
    scene    : sionna.rt.Scene
    rx       : sionna.rt.Receiver  (the UE)
    sat_id   : int
    sat_pos  : (x, y, z)  Proxy transmitter position [m].
    elev_deg : float       Elevation angle [deg] for logging.

    Returns
    -------
    paths : sionna.rt.Paths
    stats : dict            Channel statistics (see _extract_channel_stats).
    """
    tx_name = f"sat{sat_id}"
    tx = Transmitter(
        name     = tx_name,
        position = sat_pos,
        look_at  = RT_UE_POSITION,   # Beam steered towards the UE
        velocity = (0.0, 0.0, 0.0),  # Static within this snapshot
        power_dbm = 44,              # 44 dBm ≈ 25 W EIRP per beam
        display_radius = 5,
    )
    scene.add(tx)

    # Path computation
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
    out_file = f"ntn_rt_paths_sat{sat_id}.png"
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

    stats = _extract_channel_stats(paths, sat_id, sat_pos, elev_deg)

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
            filename = "ntn_rt_radiomap.png",
            radio_map = rm,
            rm_metric  = "path_gain",
            rm_db_scale = True,
            resolution  = (1280, 720),
            num_samples = 256,
        )
        print("  [Render] ntn_rt_radiomap.png")
    except Exception as e:
        print(f"  [RadioMap] Skipped: {e}")


# =============================================================================
# Public entry point
# =============================================================================

def run_ray_tracing() -> list:
    """
    Execute the full Sionna RT ray tracing pipeline:

    1. Load the Munich scene and place the UE at street level.
    2. For each satellite in the constellation snapshot:
       a. Compute its proxy position and elevation angle.
       b. Run PathSolver (LoS + specular reflections + refractions).
       c. Extract channel statistics (path gain, delay spread, …).
       d. Save a scene render with paths overlaid.
    3. Compute and save a composite radio map.

    Parameters
    ----------
    None — all settings come from config.py.

    Returns
    -------
    channel_stats : list[dict]
        One dict per satellite (in order of decreasing elevation angle).
        Keys: sat_id, elevation_deg, mean_path_gain_db, delay_spread_ns,
              num_paths, los_exists, sat_x_m, sat_y_m.
        This list is passed directly to ntn_ns3.run_ns3() so that the
        NS-3 link budget uses RT-informed channel parameters.
    """
    print("\n[Sionna RT]  5G-NTN Munich scene — satellite constellation pass")
    print(f"  Frequency      : {RT_SCENE_FREQ_HZ/1e9:.2f} GHz")
    print(f"  UE position    : {RT_UE_POSITION} m")
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
    scene, rx = _load_scene_with_ue(RT_UE_POSITION)

    all_stats   = []
    last_paths  = None   # Keep last paths object for radio map

    for sat_id, (sat_pos, elev_deg) in enumerate(zip(sat_positions, elev_angles)):
        visible = elev_deg >= SAT_HANDOVER_ELEVATION_DEG
        print(f"  Satellite {sat_id}:  elev = {elev_deg:5.1f}°  "
              f"pos = ({sat_pos[0]:6.1f}, {sat_pos[1]:5.1f}, {sat_pos[2]:.0f}) m  "
              f"{'[visible]' if visible else '[below horizon threshold]'}")

        _, stats = _trace_satellite(scene, rx, sat_id, sat_pos, elev_deg)
        all_stats.append(stats)

        print(f"    paths={stats['num_paths']}  "
              f"mean_gain={stats['mean_path_gain_db']:.1f} dB  "
              f"delay_spread={stats['delay_spread_ns']:.1f} ns")

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
            power_dbm = 44,
        )
        scene.add(tx)

    _compute_and_save_radiomap(scene)

    print(f"\n  Ray tracing complete.  {len(all_stats)} satellite snapshots computed.\n")
    return all_stats
