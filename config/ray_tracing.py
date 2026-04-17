"""
config/ray_tracing.py — Ray tracing (Sionna RT) parameters
"""

from config.phy import CARRIER_FREQ_HZ

RT_SCENE_NAME = "munich"
"""
Sionna RT built-in scene used for ray tracing.

- Must be the attribute name of a scene exposed by ``sionna.rt.scene``
  (e.g. "munich", "etoile", "florence", "simple_street_canyon").
- Swap this to change the urban environment without touching sim/ray_tracing.py.
- Custom Mitsuba XML scenes can be loaded by pointing this at the file path
  instead of a built-in name; the loader falls back to ``load_scene(path)``.
"""

RT_SCENE_FREQ_HZ = CARRIER_FREQ_HZ
"""
Carrier frequency used inside the Sionna RT scene [Hz].
- Should match CARRIER_FREQ_HZ so that material EM properties and
  wavelength-dependent diffraction are consistent with the PHY layer.
"""

RT_MAX_DEPTH = 5
"""
Maximum number of ray-surface interactions (reflections / refractions)
per traced path.
- Higher values capture more multipath components at the cost of
  exponentially increasing computation time.
- 5 is a good trade-off for urban canyons.
"""

RT_SAMPLES_PER_TX = 10 ** 6
"""
Number of rays launched per transmitter for the radio map solver.
- More samples reduce noise in the coverage map but increase runtime.
- 1 million is the Sionna RT documentation default.
"""

RT_CELL_SIZE = [1, 1]
"""
Radio map grid cell size [m, m] (x, y).
- Finer cells (e.g. [0.5, 0.5]) give higher-resolution coverage maps
  but increase memory usage proportionally to 1/cell_area.
"""

RT_UE_POSITION = [50.0, 80.0, 1.5]
"""
Representative default UE (phone) position [x, y, z] in metres within
the Munich scene.

- z = 1.5 m represents a hand-held device at typical street level.
- x=50, y=80 places the UE in a street canyon surrounded by buildings.
- This point is also included in RT_UE_SAMPLE_POSITIONS.
"""

RT_UE_SAMPLE_POSITIONS = [
    [50.0, 80.0, 1.5],
    [120.0, 30.0, 1.5],
    [-20.0, 140.0, 1.5],
    [180.0, -40.0, 1.5],
]
"""
Representative UE sampling points [x, y, z] in the Munich scene for RT.

- Instead of extracting channel stats from only one street location, the RT
  stage traces all points in this list and aggregates per-satellite channel
  statistics across them.
- This improves realism for urban environments where canyons, blockages, and
  facade materials vary significantly over short distances.
"""

RT_SAT_INITIAL_ZENITH_DEG = 20.0
"""
Zenith angle [degrees] of the first (highest-elevation) satellite
in the ray-tracing snapshot.
- Setting this > 0 avoids placing a transmitter directly overhead
  (zenith = 0°), which returns 0 valid paths in the Munich scene
  because there are no vertical surfaces to reflect near-vertical rays
  down to a street-level UE.
- 20° gives a small but non-zero horizontal offset so that building
  walls are illuminated at a glancing angle, enabling reflections.
"""

RT_SAT_SCENE_HEIGHT_M = 300.0
"""
Height [m] at which the satellite transmitters are placed *within the
Sionna RT scene* for ray tracing.
- The actual orbital altitude (550 km) is far outside the scene bounds
  (~100 m tall buildings).  This parameter places a proxy transmitter
  high above the scene to produce near-vertical incidence angles that
  are representative of a satellite link.
- The true free-space path loss over 550 km is applied analytically in
  the NS-3 link budget; only the urban multipath statistics (delay
  spread, shadow fading) are extracted from RT.
- 300 m gives a 60–70° elevation angle over a 100 m scene radius, which
  is geometrically consistent with SAT_HANDOVER_ELEVATION_DEG.
"""

RT_TX_POWER_DBM = 44.0
"""
Proxy satellite transmitter power [dBm] used in Sionna RT scenes.
- 44 dBm = 25 W EIRP per spot beam, consistent with LEO satellite
  service-link downlink power budgets (3GPP TR 38.821 §6.1 reference).
- This value is used only for the RT proxy TX in rt_sim.py; it does not
  affect the analytical link budget in ntn_ns3.py (which uses
  PHONE_EIRP_DBM / GNB_EIRP_DBM for the uplink service link and
  RT_TX_POWER_DBM for the service link).
- The absolute RT path gains are not used directly for PER calculation;
  only the *relative* gain differences between satellites are meaningful.
Source: [3GPP-38.821] §6.1, Table 6.1.1-1 (LEO sat downlink EIRP range).
"""

RT_CAM_POSITION = [-170, -170, 140]
"""Camera position [x, y, z] in metres for scene renders."""

RT_CAM_LOOK_AT = [50, 50, 0]
"""Look-at target [x, y, z] in metres for scene renders."""

RT_RENDER_PATHS = True
"""
Render per-satellite path visualisation images during ray tracing.

Set to False to skip all render_to_file calls inside _trace_satellite
(one per satellite × num_samples rays each).  On LLVM JIT these renders
are typically the slowest part of the RT stage — disabling them gives a
significant speedup at the cost of not producing ntn_rt_paths_sat<N>.png.
The radiomap render (ntn_rt_radiomap.png) is always produced regardless.
"""

RT_RENDER_NUM_SAMPLES = 64
"""
Path-tracing samples per pixel for RT scene renders.

Lower values produce noisier images but render faster.
64 gives visually acceptable quality and is 4× faster than the
reference value of 256.  Applies to both path renders and the radiomap.
"""
