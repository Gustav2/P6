"""
config.py — Shared simulation parameters
=========================================
All tunable constants are defined here and imported by the individual
simulation modules (ntn_phy.py, ntn_ns3.py, rt_sim.py, main.py).
Change values here to reconfigure the entire simulation without
touching any other file.
"""

# =============================================================================
# PHY / channel parameters  (Sionna 1.2.1 + OpenNTN TR38.811)
# =============================================================================

CARRIER_FREQ_HZ = 3.5e9
"""
5G NR carrier frequency [Hz].
- 3.5 GHz (n78) is the primary mid-band 5G-NTN band specified in
  3GPP TR 38.821 for LEO satellite service links.
- Affects OFDM wavelength, path loss, and Doppler calculations.
- Other valid choices: 2.0e9 (n255, S-band NTN), 26.5e9 (Ka-band feeder).
"""

SUBCARRIER_SPACING = 15e3
"""
OFDM subcarrier spacing [Hz].
- 15 kHz = NR numerology µ=0, standard for sub-6 GHz.
- Must be increased for mmWave (30 kHz or 60 kHz) or high-Doppler NTN
  links (120 kHz for LEO, per 3GPP TR 38.821 §6.1.2).
"""

FFT_SIZE = 128
"""
OFDM FFT size (number of subcarriers including guard bands).
- Together with SUBCARRIER_SPACING determines the OFDM symbol bandwidth.
- Bandwidth = FFT_SIZE × SUBCARRIER_SPACING = 1.92 MHz at defaults.
"""

NUM_OFDM_SYMBOLS = 14
"""Number of OFDM symbols per slot (1 NR slot at µ=0)."""

CP_LENGTH = 9
"""
Cyclic prefix length [samples].
- Must exceed the maximum multipath delay spread to avoid ISI.
- 9 samples × (1/15 kHz) ≈ 4.7 µs; sufficient for urban delay spreads
  of ~1 µs but marginal for long-delay NTN multipath.
"""

PILOT_SYMBOL_IDX = [2, 11]
"""
OFDM symbol indices used for pilot (channel estimation) symbols.
- Two pilots per slot following 3GPP NR DMRS Type 1, positions 2 and 11.
"""

NUM_BITS_PER_SYMBOL = 2
"""
Bits per modulation symbol.
- 2 = QPSK. Change to 4 (16-QAM) or 6 (64-QAM) to test higher orders.
- Trades spectral efficiency against robustness to noise/fading.
"""

CODERATE = 0.5
"""
LDPC channel code rate (k/n).
- 0.5 = half-rate; 50 % overhead for error correction.
- Lower rates (e.g. 0.33) improve link reliability at the cost of
  reduced information throughput.
"""

BATCH_SIZE = 64
"""
Number of independent channel realisations evaluated per SNR point.
- Higher values reduce Monte-Carlo variance but increase memory/time.
"""

# =============================================================================
# Satellite orbit and constellation parameters
# =============================================================================

SAT_HEIGHT_M = 600_000.0
"""
Nominal LEO satellite orbital altitude [m].
- 600 km is representative of Starlink-like LEO constellations.
- Affects one-way propagation delay (~2 ms/600 km slant range) and
  free-space path loss.
- OpenNTN expects this value divided by 1000 (i.e. in km).
"""

ELEVATION_ANGLE_DEG = 60.0
"""
Minimum UE-to-satellite elevation angle [degrees] used for the PHY
channel model constructor.  The RT simulation sweeps through multiple
elevation angles as the constellation moves overhead.
"""

SAT_ORBIT_INCLINATION_DEG = 53.0
"""
Orbital inclination of the satellite constellation [degrees].
- 53° matches SpaceX Starlink shell 1.
- Affects the ground track and how quickly satellites pass overhead.
"""

SAT_ORBITAL_VELOCITY_MS = 7600.0
"""
Orbital velocity of a LEO satellite at SAT_HEIGHT_M [m/s].
- Derived from v = sqrt(GM / (R_E + h)):
  sqrt(3.986e14 / (6.371e6 + 600e3)) ≈ 7558 m/s ≈ 7600 m/s.
- Causes Doppler shifts of up to ±24 kHz at 3.5 GHz (v/c × f).
"""

NUM_SATELLITES = 3
"""
Number of satellites in the simulated constellation pass.
- Each satellite is visible for a limited time window; when one drops
  below MIN_ELEVATION_DEG the UE performs a handover to the next one.
- Minimum 2 to observe at least one handover event.
"""

SAT_HANDOVER_ELEVATION_DEG = 10.0
"""
Elevation angle threshold [degrees] at which a handover is triggered.
- When the serving satellite falls below this angle, the UE connects
  to the next satellite that has risen above this threshold.
- 10° is typical for LEO NTN systems (3GPP TR 38.821 §6.1).
"""

SAT_SPACING_DEG = 15.0
"""
Angular spacing between consecutive satellites along the orbit [degrees].
- Determines how long the gap between satellites is (and therefore
  whether seamless handover is possible or a link interruption occurs).
- At 600 km altitude, 15° ≈ 157 km ground-track separation.
"""

# =============================================================================
# Ray tracing (Sionna RT) parameters
# =============================================================================

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
UE (phone) position [x, y, z] in metres within the Munich scene.
- z = 1.5 m represents a hand-held device at typical street level.
- x=50, y=80 places the UE in a street canyon surrounded by buildings
  so that reflected/diffracted paths from all satellite directions
  can be resolved by the ray tracer (verified: 4/4/8 paths for
  satellites at zenith 20°/35°/50°).
- Scene bounding box: x ∈ [−806, 670] m, y ∈ [−689, 517] m.
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
- The actual orbital altitude (600 km) is far outside the scene bounds
  (~100 m tall buildings).  This parameter places a proxy transmitter
  high above the scene to produce near-vertical incidence angles that
  are representative of a satellite link.
- The true free-space path loss over 600 km is applied analytically in
  the NS-3 link budget; only the urban multipath statistics (delay
  spread, shadow fading) are extracted from RT.
- 300 m gives a 60–70° elevation angle over a 100 m scene radius, which
  is geometrically consistent with SAT_HANDOVER_ELEVATION_DEG.
"""

RT_CAM_POSITION = [-170, -170, 140]
"""Camera position [x, y, z] in metres for scene renders."""

RT_CAM_LOOK_AT = [50, 50, 0]
"""Look-at target [x, y, z] in metres for scene renders."""

# =============================================================================
# NS-3 network simulation parameters
# =============================================================================

SIM_DURATION_S = 60.0
"""
Total NS-3 simulation duration [seconds].
- Long enough to capture multiple satellite handovers.
  At 7600 m/s and 600 km altitude, a full overhead pass takes ~8 min;
  60 s shows at least one handover with NUM_SATELLITES = 3.
"""

# =============================================================================
# Protocol comparison settings
# =============================================================================

PROTOCOLS = [
    {"protocol": "udp",  "tcp_variant": None,       "label": "UDP"},
    {"protocol": "tcp",  "tcp_variant": "NewReno",   "label": "TCP NewReno"},
    {"protocol": "tcp",  "tcp_variant": "Cubic",     "label": "TCP CUBIC"},
    {"protocol": "tcp",  "tcp_variant": "Bbr",       "label": "TCP BBR"},
]
"""
List of transport protocol configurations to simulate and compare.
Each entry is a dict with:
  protocol    : "udp" | "tcp"
                UDP uses a constant-bit-rate OnOff application.
                TCP uses a BulkSend application (saturating sender).
  tcp_variant : NS-3 TCP congestion-control algorithm name, or None.
                Valid values: "NewReno", "Cubic", "Bbr", "Westwood", "Veno".
                Ignored when protocol is "udp".
  label       : Human-readable name used in plots and console output.
"""

APP_DATA_RATE = "5Mbps"
"""
Application-layer target data rate for UDP CBR traffic [NS-3 string].
- Used only for the UDP OnOff application.
- TCP (BulkSend) always attempts to fill the pipe; this value is ignored
  for TCP runs.
- Format: "<number><unit>" where unit is bps / kbps / Mbps / Gbps.
"""

PACKET_SIZE_BYTES = 1400
"""
Application payload size per packet [bytes].
- 1400 bytes avoids IP fragmentation on a 1500-byte MTU Ethernet path.
- For NTN links with high BDP, smaller packets (e.g. 500 B) increase
  overhead; larger packets (1400 B) are more efficient.
"""

NUM_PARALLEL_FLOWS = 1
"""
Number of simultaneous application flows per protocol run.
- Set to > 1 to study the effect of flow multiplicity on congestion
  control (e.g. TCP CUBIC with 4 flows vs. BBR with 4 flows).
"""

TCP_SNDRCV_BUF_BYTES = 512_000
"""
TCP send and receive socket buffer size [bytes].
- Must be at least BDP = RTT × bandwidth to avoid throughput starvation.
- At 600 km LEO (RTT ≈ 36 ms round-trip through sat+feeder+fibre ≈ 56 ms)
  and 10 Mbps: BDP ≈ 70 kB.  512 kB gives ~7× headroom.
- Both snd and rcv buffers are set to this value.
"""

TCP_SACK_ENABLED = True
"""
Enable TCP Selective Acknowledgements (SACK, RFC 2018).
- Strongly recommended for satellite links where multiple segments may
  be lost in a single window; SACK avoids go-back-N retransmissions.
- Set to False to simulate legacy stacks without SACK.
"""

TCP_TIMESTAMPS = True
"""
Enable TCP Timestamps option (RFC 7323).
- Enables more accurate RTT measurement and PAWS (Protection Against
  Wrapped Sequence numbers), both beneficial on long-delay NTN paths.
- Set to False to reduce per-packet header overhead (10 bytes/packet).
"""
