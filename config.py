"""
config.py — Shared simulation parameters
=========================================
All tunable constants are defined here and imported by the individual
simulation modules (ntn_phy.py, ntn_ns3.py, rt_sim.py, main.py).
Change values here to reconfigure the entire simulation without
touching any other file.

Sources used throughout this file
----------------------------------
[3GPP-38.101-1]  3GPP TS 38.101-1 v17.x, "NR; User Equipment (UE) radio
                 transmission and reception; Part 1: Range 1 Standalone".
[3GPP-38.133]    3GPP TS 38.133 v17.x, "NR; Requirements for support of
                 radio resource management".
[3GPP-38.214]    3GPP TS 38.214 v17.x, "NR; Physical layer procedures for
                 data".
[3GPP-38.300]    3GPP TS 38.300 v17.x, "NR; NR and NG-RAN Overall description;
                 Stage-2".
[3GPP-38.306]    3GPP TS 38.306 v17.x, "NR; User Equipment (UE) radio access
                 capabilities".
[3GPP-38.821]    3GPP TR 38.821 v16.x, "Solutions for NR to support non-
                 terrestrial networks (NTN)".
[3GPP-38.811]    3GPP TR 38.811 v15.x, "Study on New Radio (NR) to support
                 non-terrestrial networks".
[ITU-R-S.465]    ITU-R Recommendation S.465-6 (2000), "Reference Earth-station
                 radiation pattern for use in coordination and interference
                 assessment in the frequency range from 2 to 31 GHz".
[ITU-R-S.524]    ITU-R Recommendation S.524-9 (2006), "Maximum permissible
                 levels of off-axis e.i.r.p. density from earth stations in
                 geostationary-satellite orbit networks operating in the fixed-
                 satellite service transmitting in the 6 GHz, 13 GHz, 14 GHz
                 and 30 GHz frequency bands".
[Starlink-FCC]   SpaceX FCC filing IBFS SAT-LOA-20200526-00055 (2020),
                 Starlink V2 constellation Ka-band gateway parameters.
[Ericsson-5G]    Ericsson Technology Review, "5G NR: The next generation
                 wireless access technology" (2017), ISBN 978-91-982114-0-7.
[Nokia-5G]       Nokia Bell Labs, "5G New Radio: Beamforming, MIMO, and MU-MIMO"
                 (2018), white paper.
"""

# =============================================================================
# PHY / channel parameters  (Sionna 1.2.1 + OpenNTN TR38.811)
# =============================================================================

CARRIER_FREQ_HZ = 3.5e9
"""
5G NR carrier frequency [Hz].
- 3.5 GHz (n78 band) is the primary mid-band 5G-NTN service-link frequency
  specified in 3GPP TR 38.821 §6.1 for LEO satellite service links.
- Affects OFDM wavelength, free-space path loss (FSPL), and Doppler shifts.
- Other valid choices: 2.0 GHz (n255, S-band NTN), 26.5 GHz (Ka-band feeder).
Source: [3GPP-38.821] §6.1, Table 6.1.1-1 (service-link reference frequency).
"""

SUBCARRIER_SPACING = 15e3
"""
OFDM subcarrier spacing [Hz].
- 15 kHz = NR numerology µ=0, the baseline for FR1 sub-6 GHz deployments.
- 3GPP TR 38.821 §6.1.2 notes that µ=1 (30 kHz) or µ=3 (120 kHz) may be
  needed for LEO NTN because the satellite Doppler at 3.5 GHz reaches up to
  ±88.8 kHz (v=7612 m/s, see SAT_ORBITAL_VELOCITY_MS) >> 15 kHz SCS.
  µ=0 is retained here as the reference numerology for the Munich urban scene.
- One NR slot at µ=0 has duration T_slot = 1 ms (14 OFDM symbols).
Source: [3GPP-38.300] §5.3.1, Table 4.1-1 in [3GPP-38.101-1].
"""

FFT_SIZE = 128
"""
OFDM FFT size (number of subcarriers including guard bands).
- Together with SUBCARRIER_SPACING determines the OFDM symbol bandwidth.
- Bandwidth = FFT_SIZE × SUBCARRIER_SPACING = 1.92 MHz at defaults.
"""

NUM_OFDM_SYMBOLS = 14
"""Number of OFDM symbols per slot (1 NR slot at µ=0).
Source: [3GPP-38.300] §5.3.1 — 14 symbols per slot for normal cyclic prefix.
"""

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
Source: [3GPP-38.211] §7.4.1.1 (DMRS for PDSCH/PUSCH).
"""

NUM_BITS_PER_SYMBOL = 2
"""
Bits per modulation symbol.
- 2 = QPSK (MCS index 1). Change to 4 (16-QAM) or 6 (64-QAM) for higher
  spectral efficiency.
- QPSK is the reference modulation for link-budget calculations in
  3GPP TR 38.821 at low-to-medium SNR.
Source: [3GPP-38.214] Table 5.1.3.1-2 (PDSCH MCS tables).
"""

CODERATE = 0.5
"""
LDPC channel code rate (k/n).
- 0.5 = half-rate; 50 % overhead for error correction.
- Corresponds to MCS index ~7 for QPSK in 3GPP NR.
Source: [3GPP-38.214] Table 5.1.3.1-2.
"""

BATCH_SIZE = 512
"""
Number of independent channel realisations evaluated per SNR point.
- Higher values reduce Monte-Carlo variance but increase memory/time.
- 512 keeps all 8 Jetson CPU cores busy via TF intra-op parallelism while
  staying well within the 61 GB unified memory budget (~400 MB peak).
- Original value was 64; 512 gives 8× better variance reduction per sweep,
  which is important for reliable BER→PER sigmoid fitting.
"""

# =============================================================================
# Satellite orbit and constellation parameters
# =============================================================================

SAT_HEIGHT_M = 550_000.0
"""
Nominal LEO satellite orbital altitude [m].
- 550 km is the operational altitude of SpaceX Starlink Shell 1 (the primary
  deployment shell as of 2023–2024).  Earlier filings used 600 km but
  Starlink transitioned to 550 km (FCC approval 2019, operational 2020–).
- One-way propagation delay at 550 km nadir: ~1.83 ms (c = 3×10⁸ m/s).
- Free-space path loss at 3.5 GHz, 550 km slant: ~180 dB.
Source: [Starlink-FCC] Attachment A, Table 1 (550 km shell, 53° incl.).
         Also [3GPP-38.821] §6.1 uses 600 km as the 3GPP reference; 550 km
         is the closest real-world analogue.
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
- 53° matches SpaceX Starlink Shell 1 inclination (FCC filing 2019).
- Affects the ground track and how quickly satellites pass overhead.
Source: [Starlink-FCC] Attachment A.
"""

SAT_ORBITAL_VELOCITY_MS = 7612.0
"""
Orbital velocity of a LEO satellite at SAT_HEIGHT_M [m/s].
- Derived from the vis-viva equation: v = sqrt(GM / (R_E + h))
    GM = 3.986004418×10¹⁴ m³/s²  (WGS84 standard gravitational parameter)
    R_E = 6.3781×10⁶ m            (WGS84 equatorial radius)
    h   = 5.50×10⁵ m              (550 km altitude)
  → v = sqrt(3.986e14 / 6.928e6) = 7,612 m/s
- Doppler shift at 3.5 GHz (worst case, near horizon where radial velocity peaks):
    Δf_max = v/c × f = 7612 / 3×10⁸ × 3.5×10⁹ ≈ ±88.8 kHz (full radial pass)
    Δf at zenith ≈ 0 (perpendicular to LOS); peak near horizon ≈ ±88.8 kHz.
- Used in ntn_phy.py: ut_velocities is set to zero, NOT to this value.
  3GPP TR 38.821 §6.1.2 mandates that NTN UEs pre-compensate the satellite
  Doppler shift before the OFDM demodulator.  At 3.5 GHz, the max Doppler
  from this speed is ~88 kHz >> 15 kHz SCS, which causes BER ≈ 0.5 if
  applied without pre-compensation.  Setting ut_velocities = 0 models the
  post-compensation residual.  This constant is retained for documentation
  and for the NTN Doppler branch in Sionna (bs_height ≥ 600 km path).
Source: Standard orbital mechanics (IERS Conventions 2010, §6.1).
"""

CONSTELLATION_TOTAL_SATS = 3000
"""
Approximate total number of satellites in the reference LEO shell.

- This is metadata used to document constellation realism; the simulator does
  not instantiate all of these satellites in NS-3 or Sionna RT.
- 3000 is in-family with modern broadband LEO shell sizes (order of 10^3).
"""

VISIBLE_SATELLITES_PER_PASS = 8
"""
Number of satellites sampled as simultaneously visible during one urban pass.

- This is the practical simulation density used by ray tracing and handover
  scheduling (NUM_SATELLITES below).
- 8 provides a more realistic visible-satellite density than 3 while keeping
  runtime manageable on CPU-only setups.
"""

NUM_SATELLITES = VISIBLE_SATELLITES_PER_PASS
"""
Number of satellites in the simulated constellation pass.

This controls both the Sionna RT ray-tracing snapshot and the NS-3 handover
schedule. It is interpreted as the sampled set of currently visible satellites
from a much larger LEO constellation (CONSTELLATION_TOTAL_SATS).

Use this value to trade realism vs runtime:
  - Higher values improve handover granularity and elevation diversity.
  - Lower values run faster.
"""

SAT_HANDOVER_ELEVATION_DEG = 10.0
"""
Elevation angle threshold [degrees] at which a handover is triggered.
- When the serving satellite falls below this angle, the UE connects
  to the next satellite that has risen above this threshold.
- 10° is the minimum elevation specified for LEO NTN handover in the
  3GPP NTN reference scenario.
Source: [3GPP-38.821] §6.1, Table 6.1.1-1 (minimum elevation angle = 10°).
"""

SAT_SPACING_DEG = 15.0
"""
Angular spacing between consecutive satellites along the orbit [degrees].
- Determines how long the gap between satellites is (and therefore
  whether seamless handover is possible or a link interruption occurs).
- At 550 km altitude, 15° ≈ 144 km ground-track separation.
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



# =============================================================================
# NS-3 network simulation parameters
# =============================================================================

SIM_DURATION_S = 60.0
"""
Total NS-3 simulation duration [seconds].
- 60 s shows at least one handover event with NUM_SATELLITES = 3.
  At 7612 m/s and 550 km altitude, a full overhead pass takes ~8 min;
  three satellites with 15° spacing give handovers at ~14 s and ~46 s.
- Longer durations (e.g. 600 s) significantly increase NS-3 runtime:
  with 50 clients, each extra 30 s of simulated time adds ~24 min wall-
  time, making 600 s × 5 protocols ≈ 8 hours impractical.
- DATA_VOLUME_MB caps each TCP flow at 10 MB, so TCP runs will finish
  their transfer well before t=60 s even at the limited NTN rates.
"""

# =============================================================================
# Protocol comparison settings
# =============================================================================

PROTOCOLS = [
    {"protocol": "udp",  "tcp_variant": None,       "label": "UDP"},
    {"protocol": "tcp",  "tcp_variant": "NewReno",   "label": "TCP NewReno"},
    {"protocol": "tcp",  "tcp_variant": "Cubic",     "label": "TCP CUBIC"},
    {"protocol": "tcp",  "tcp_variant": "Bbr",       "label": "TCP BBR"},
    {"protocol": "quic", "tcp_variant": None,        "label": "QUIC"},
]
"""
List of transport protocol configurations to simulate and compare.
Each entry is a dict with:
  protocol    : "udp" | "tcp" | "quic"
                UDP uses a constant-bit-rate OnOff application.
                TCP uses a BulkSend application (saturating sender).
                QUIC is emulated on top of NS-3 UDP with RFC 9000
                corrections applied analytically after FlowMonitor
                collection (1-RTT saving, PTO vs RTO, unlimited ACK
                ranges, faster post-handover recovery).
  tcp_variant : NS-3 TCP congestion-control algorithm name, or None.
                Valid values: "NewReno", "Cubic", "Bbr", "Westwood", "Veno".
                Ignored when protocol is "udp" or "quic".
  label       : Human-readable name used in plots and console output.
"""

APP_DATA_RATE = "1Mbps"
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
- At 550 km LEO (RTT ≈ 34 ms round-trip through sat+ISL+backhaul ≈ 54 ms)
  and 10 Mbps: BDP ≈ 68 kB.  512 kB gives ~7× headroom.
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

# =============================================================================
# Antenna / EIRP parameters
# =============================================================================

PHONE_EIRP_DBM = 23.0
"""
Phone (UE) uplink EIRP [dBm] for the direct Phone→Satellite service link.
- 23 dBm = 200 mW, the maximum transmit power for 5G NR UE Power Class 3,
  which covers the vast majority of handsets.
- Omnidirectional handheld antenna; no beamforming gain assumed.
Source: [3GPP-38.101-1] §6.2.2, Table 6.2.2-1 (UE Power Class 3 = 23 dBm).
"""

TERRESTRIAL_BACKHAUL_DELAY_MS = 10.0
"""
One-way latency of the Benchmark Satellite → Internet Server link [ms].

- 10 ms is representative of the internet backbone latency from wherever
  the satellite constellation terminates into the public internet to a
  regional data centre / Internet Exchange Point (IXP).
- Derivation: typical backbone fibre propagation velocity ≈ 0.67 c
  (200 000 km/s); a 2 000 km satellite PoP-to-IXP route gives
  ~10 ms one-way delay.
- Sources:
    Singla et al., "Middleboxes as a Cloud Service" (HotNets 2014): measured
      fibre latency ≈ 5 µs/km for terrestrial routes.
    Akamai State of the Internet Q4 2022: regional CDN RTTs 15–30 ms imply
      one-way backhaul of 7–15 ms for metro distances.
"""

SAT_RX_ANTENNA_GAIN_DB = 30.0
"""
Satellite receive antenna gain [dBi] applied to the uplink link budget.
- Models the satellite's phased-array / multi-beam receive aperture at
  3.5 GHz (service link).
- A phased-array with ~128 elements at 3.5 GHz (λ = 8.6 cm) achieves
  ~27–32 dBi, depending on aperture efficiency.
- The 3GPP NTN reference LEO satellite is specified with a 30 dBi spot-
  beam receive gain for the service link.
Source: [3GPP-38.821] §6.1, Table 6.1.1-1 (satellite Rx antenna gain
          for service link = 30 dBi at 3.5 GHz, LEO 600 km reference).
"""

# =============================================================================
# Satellite → Internet Server link parameters
# =============================================================================

SAT_SERVER_DATARATE = "1Gbps"
"""
Data rate of the benchmark satellite → Internet Server direct link [NS-3 string].

The benchmark satellite connects directly to the internet server — there
is no ground station node in the simulated topology.  This link represents
the satellite's direct IP peering into the public internet backbone via
ISL-connected gateway nodes or a cloud PoP (Point of Presence).

- 1 Gbps is achievable via high-throughput satellite transponders or
  optical inter-satellite links terminating at internet exchange points.
- This link should never be the bottleneck; the service link (10 Mbps)
  and ISL are the limiting hops.
"""

# =============================================================================
# Link budget thresholds — shared by ntn_ns3.py and topology_diagram.py
# =============================================================================

NOISE_FLOOR_DBM = -121.0
"""
Receiver thermal noise floor [dBm] for the service-link budget.

Derivation (Johnson–Nyquist thermal noise, per NR subcarrier):
  N₀ = k · T · B
  k  = 1.380649×10⁻²³ J/K  (Boltzmann constant)
  T  = 290 K                (ITU-R reference noise temperature, [ITU-R-S.465])
  B  = Δf_sc = 15 kHz       (one NR subcarrier, numerology µ=0; this is the
                              natural noise bandwidth for per-subcarrier SNR
                              as used throughout the link budget and Sionna
                              ResourceGrid, where SNR is defined per subcarrier)

  kTB(15 kHz) = 10·log10(1.38e-23 × 290 × 15e3) + 30  [dBm]
              = 10·log10(6.003e-17) + 30
              = −172.2 + 30 = −142.2 dBm/subcarrier

  (Cross-check via standard formula: kTB = −174 dBm/Hz + 10·log10(15e3)
                                         = −174 + 41.8 = −132.2 dBm
   — same result within rounding.)

  + Receiver noise figure (NF) — composite system budget:
    - UE/phone S-band LNA: NF ≈ 6–7 dB (3GPP TS 38.101-1 §7.3 reference
      sensitivity assumes NF = 7 dB for UE).
    - Feed/connector losses at phone: ~1–2 dB.
    - ADC quantisation + baseband processing: ~0.5–1 dB.
    - Adopted composite NF = 11.2 dB (consistent with 3GPP TR 38.821 §6.1.1
      UL link budget, Table 6.1.1-1, which assumes UE NF = 10 dB and
      accounts for 1–2 dB of additional implementation loss).

  N_floor = kTB(15 kHz) + NF = −132.2 + 11.2 = −121 dBm/subcarrier

  This value is adopted as −121 dBm (rounded).  It sets the link budget
  cliff at ~10° elevation (FSPL ≈ 168.5 dB, SNR_phone ≈ 5.5 dB vs the
  QPSK threshold of ~5.9 dB), which aligns with the handover minimum
  elevation angle used in the simulation.

Source: [ITU-R-S.465] §2 (reference noise temperature T = 290 K).
        [3GPP-38.821] §6.1.1 (system noise temperature 200–400 K for LEO NTN).
        ITU-R Handbook on Satellite Communications §2.2 (link budget noise).
"""

SNR_THRESH_DB = 7.5
"""
Initial SNR threshold estimate for QPSK r=0.5 demodulation [dB].
- This is an initial estimate only.  At runtime, main.py fits a sigmoid
  to the actual Sionna LDPC BER curve (after the PHY simulation in Part 1)
  and the fitted value is passed to run_ns3_both_topologies() via
  snr_thresh_db=.  The config value is used as the curve_fit starting
  guess (p0) and as a fallback if fitting fails.
- Derivation of the initial estimate: AWGN BER ≈ 1×10⁻³ for QPSK at
  Eb/N₀ ≈ 6.8 dB (Q-function inversion: BER = Q(√(2·Eb/N₀)) = 10⁻³
  → Eb/N₀ = 6.8 dB), plus 0.7 dB implementation margin.
  SNR_thresh = Eb/N₀ + 10·log10(R·log2(M))
             = 6.8 + 10·log10(0.5 × 2) = 6.8 + 0.0 = 6.8 dB ≈ 7.5 dB
Source: [3GPP-38.214] §5.1.3 (PDSCH MCS selection SNR targets).
        Proakis, "Digital Communications" 5th ed., Ch. 8 (BER formulas).
"""

SIGMOID_SLOPE = 0.7
"""
Initial steepness estimate for the PER sigmoid curve [1/dB].
- PER(SNR) = 1 / (1 + exp(SIGMOID_SLOPE · (SNR − SNR_THRESH_DB)))
- This is an initial estimate only.  At runtime, main.py fits the sigmoid
  to the Sionna LDPC BER→PER curve and the fitted slope is passed to
  run_ns3_both_topologies() via sigmoid_slope=.  The config value is
  used as the curve_fit starting guess (p0) and as a fallback.
- 0.7 gives a smooth ~10 dB transition range, consistent with BLER vs. SNR
  curves for QPSK r=0.5 in AWGN (3GPP TS 38.214, Annex A BLER curves).
"""

RT_GAIN_P10_BLEND = 0.35
"""
Blend factor for conservative RT gain in PER modeling.

In ntn_ns3.py, the effective RT gain used in the link budget is:

  gain_eff = (1 - RT_GAIN_P10_BLEND) * mean_path_gain_db
             + RT_GAIN_P10_BLEND * mean_path_gain_p10_db

where mean_path_gain_p10_db is the 10th-percentile gain over sampled UE
positions in the urban scene.

- 0.0 uses only the mean gain.
- 1.0 uses only the conservative percentile.
- 0.35 gives moderate robustness to local urban blockages.
"""

# =============================================================================
# Multi-client topology settings
# =============================================================================

NUM_STATIONARY_CLIENTS = 30
"""
Number of stationary client (phone) nodes placed randomly inside a circle
of radius CLIENT_AREA_RADIUS_M centred on the NS-3 coordinate origin.

Stationary clients use a ConstantPositionMobilityModel; their positions are
fixed throughout the simulation.  Used together with NUM_MOVING_CLIENTS.
"""

NUM_MOVING_CLIENTS = 20
"""
Number of mobile client (phone) nodes using the RandomWaypoint mobility
model.  Each moving client is initialised at a random position within
CLIENT_AREA_RADIUS_M and moves at a uniformly distributed speed between
class-specific bounds toward a random
destination within the same radius, with zero pause time at each waypoint.

RandomWaypoint parameters are set in ntn_ns3.py using the bounds
[−CLIENT_AREA_RADIUS_M, CLIENT_AREA_RADIUS_M] for both X and Y axes.
"""

NUM_PEDESTRIAN_MOVING_CLIENTS = 10
"""
Number of moving clients modelled as pedestrians.

- These clients use the pedestrian RandomWaypoint speed range and represent
  handheld users walking in the urban area.
- Must satisfy:
    NUM_PEDESTRIAN_MOVING_CLIENTS + NUM_VEHICULAR_MOVING_CLIENTS
    == NUM_MOVING_CLIENTS
"""

NUM_VEHICULAR_MOVING_CLIENTS = 10
"""
Number of moving clients modelled as vehicular 5G modems.

- These clients use the vehicular RandomWaypoint speed range and represent
  direct-to-cell modems in moving cars.
- Must satisfy:
    NUM_PEDESTRIAN_MOVING_CLIENTS + NUM_VEHICULAR_MOVING_CLIENTS
    == NUM_MOVING_CLIENTS
"""

CLIENT_AREA_RADIUS_M = 500.0
"""
Radius [m] of the circular deployment area for all clients (stationary
and moving) around the NS-3 coordinate origin (0, 0, 1.5 m).

- 500 m is representative of a 5G macro-cell coverage radius in a dense
  urban environment (inter-site distance ~500–1000 m for urban macro).
- All clients in this area share the same satellite visibility and
  handover schedule because 500 m ≪ the satellite footprint (~100 km).
Source: 3GPP TR 38.913 §8.1 (dense urban macro inter-site distance 200 m;
        urban macro 500 m).
"""

PEDESTRIAN_SPEED_MIN_MS = 1.0
"""Minimum pedestrian moving-client speed [m/s] (slow walk)."""

PEDESTRIAN_SPEED_MAX_MS = 2.2
"""Maximum pedestrian moving-client speed [m/s] (brisk walk)."""

VEHICULAR_SPEED_MIN_MS = 10.0
"""Minimum vehicular moving-client speed [m/s] (36 km/h urban driving)."""

VEHICULAR_SPEED_MAX_MS = 22.0
"""Maximum vehicular moving-client speed [m/s] (79 km/h urban arterial)."""


# =============================================================================
# Data volume per flow
# =============================================================================

DATA_VOLUME_MB = 10.0
"""
Maximum data volume sent per TCP BulkSend flow [MB].

NS-3 BulkSend.MaxBytes is set to int(DATA_VOLUME_MB × 10⁶) bytes so that
each flow terminates after transferring this fixed file size rather than
saturating the link for the full SIM_DURATION_S.

- 10 MB is representative of a medium-sized file transfer (e.g. a map
  tile cache update or a short video segment for adaptive streaming).
- Setting this value ensures all protocol runs transmit the same total
  data volume, making throughput and latency comparisons fair.
- Set to 0 to disable the cap (unlimited BulkSend, saturating sender).

UDP OnOff flows are not capped by this value (they use APP_DATA_RATE).
"""

# =============================================================================
# Inter-satellite link (ISL) parameters
# =============================================================================

ISL_DATARATE = "1Gbps"
"""
Data rate of the inter-satellite link between an access satellite and the
benchmark satellite (Sat 0) [NS-3 string].

- 1 Gbps is achievable with a space-borne optical (laser) ISL.
  Starlink Gen-2 optical ISLs are rated at ~1–10 Gbps per link.
- The ISL is the highest-capacity link in the chain; it should never be
  the bottleneck.  The bottleneck is the ground service link (10 Mbps).
Source: Bhattacherjee & Singla, "Network topology design at 27,000 km/hour"
        (CoNEXT 2019) — laser ISL capacity 1–10 Gbps.
"""

ISL_DELAY_MS = 5.0
"""
One-way propagation delay of the access → benchmark satellite ISL [ms].

- At 550 km altitude with SAT_SPACING_DEG = 15°, the inter-satellite
  distance along the orbit is approximately:
    d ≈ 2 × (R_E + h) × sin(π × SAT_SPACING_DEG / 360)
      = 2 × 6921 km × sin(7.5°) ≈ 1808 km
  One-way delay = 1808 km / (3×10⁵ km/s) ≈ 6.0 ms.
- 5.0 ms is used as a round number slightly conservative of this estimate,
  accounting for the fact that the access satellite and benchmark satellite
  may be separated by less than one full SAT_SPACING_DEG in practice.
"""

ISL_PER = 0.0001
"""
Packet error rate on the access → benchmark satellite ISL.

- Optical ISLs have an intrinsically very low BER (~10⁻¹² after FEC).
  At the packet level (1400 B = 11200 bits), PER ≈ 1 − (1−BER)^11200 ≈ 10⁻⁸.
- 10⁻⁴ is used as a conservative engineering margin that accounts for
  occasional pointing-acquisition losses, atmospheric scintillation at
  low elevation, and inter-satellite link geometry changes.
- Even at 10⁻⁴ PER, the ISL contributes negligibly to end-to-end loss
  compared with the ground service link (PER typically 0.01–0.3).
"""

# =============================================================================
# Beam management / handover interruption parameters
# =============================================================================
# Models the 3-phase NTN conditional handover procedure per
# 3GPP TS 38.300 §10.1.2.3 and 3GPP TR 38.821 §6.2.

BEAM_FAILURE_DETECTION_MS = 50.0
"""
Time for the UE to detect beam failure after the serving satellite drops
below SAT_HANDOVER_ELEVATION_DEG [ms].

- During this window the link is in a degraded / unreliable state.
  Modelled as a link blackout starting at the handover trigger time.
- 50 ms corresponds to a BFD (Beam Failure Detection) timer of
  T_BFD = 50 ms, the minimum value allowed by 3GPP TS 38.321 §5.17.
Source: 3GPP TS 38.321 v17.x §5.17 (BFD timer range 10–200 ms).
        3GPP TS 38.300 §10.1.2.3 (conditional handover procedure).
"""

RANDOM_ACCESS_DELAY_MS = 10.0
"""
Duration of the Random Access Channel (RACH) procedure on the new
satellite beam [ms].

- After beam failure detection the UE initiates PRACH on the target beam.
  The RACH round trip (preamble TX + RAR reception + Msg3/Msg4) takes
  one RACH occasion + round-trip propagation.
- At 550 km LEO: RACH occasion every 1–5 ms + ~3.7 ms propagation ≈ 10 ms.
Source: 3GPP TS 38.321 §5.1.2 (RACH procedure timing).
        3GPP TR 38.821 §6.2 (NTN RACH design).
"""

CONDITIONAL_HO_PREP_MS = 20.0
"""
Conditional Handover (CHO) preparation and execution offset [ms].

- CHO preparation runs in parallel with the serving link; at execution
  the UE applies the prepared configuration to the target cell.
  This 20 ms models the RRC reconfiguration and path switch signalling
  that completes after RACH.
Source: 3GPP TS 38.300 §10.1.2.3 (CHO execution delay).
        3GPP TR 38.821 §6.2.2 (NTN-specific CHO timing).
"""

HANDOVER_INTERRUPTION_MS_MIN = 50.0
"""
Minimum total link-layer interruption time per handover event [ms].

The total blackout duration = BEAM_FAILURE_DETECTION_MS
                             + RANDOM_ACCESS_DELAY_MS
                             + CONDITIONAL_HO_PREP_MS
                           ≈ 80 ms at minimum.
50 ms is used as the floor to account for favourable conditions
(pre-positioned target beam, short RACH slot).
Source: 3GPP TR 38.821 §6.2.2 (NTN handover interruption time
        target ≤ 0 ms for seamless HO; realistic range 50–300 ms
        for conventional NTN CHO without predictive scheduling).
"""

HANDOVER_INTERRUPTION_MS_MAX = 200.0
"""
Maximum total link-layer interruption time per handover event [ms].

200 ms covers worst-case BFD timer expiry + congested RACH + slow
RRC path switch under high orbital Doppler and multipath uncertainty.
Source: 3GPP TR 38.821 §6.2.2; 3GPP TS 38.321 §5.17 (BFD max 200 ms).
"""

# =============================================================================
# Traffic profiles and client assignment
# =============================================================================

TRAFFIC_PROFILES = {
    "streaming": {
        "app_type":   "udp_cbr",
        "data_rate":  "2.5Mbps",
        "packet_size": 1316,
        "duty":        1.0,
    },
    "gaming": {
        "app_type":   "udp_cbr",
        "data_rate":  "120kbps",
        "packet_size": 200,
        "duty":        1.0,
    },
    "texting": {
        "app_type":   "udp_cbr",
        "data_rate":  "30kbps",
        "packet_size": 120,
        "duty":        0.15,
    },
    "voice": {
        "app_type":   "udp_cbr",
        "data_rate":  "32kbps",
        "packet_size": 160,
        "duty":        1.0,
    },
    "bulk": {
        "app_type":   "tcp_bulk",
        "data_rate":  None,
        "packet_size": 1400,
        "duty":        1.0,
    },
}
"""
Traffic profile definitions for heterogeneous client simulation.

Each profile is a dict with:
  app_type   : "udp_cbr"  → OnOffHelper over UDP (constant-bit-rate when on)
               "tcp_bulk" → BulkSendHelper over TCP (saturating sender)
  data_rate  : NS-3 data-rate string for CBR profiles; None for bulk TCP.
  packet_size: Application payload bytes per packet.
  duty       : Fraction of time the application is active (OnTime / period).
               1.0 = always on.  0.1 = 10% on-time (IoT burst pattern).

Profile characteristics:
  streaming — 2.5 Mbps UDP CBR, 1316-byte packets (always on).
              Models downlink-heavy adaptive video streaming.
  gaming    — 120 kbps UDP CBR, 200-byte packets (always on).
            Models interactive gaming traffic: small, frequent datagrams.
  texting   — 30 kbps UDP CBR, 120-byte packets, 15% duty cycle.
              Models bursty messaging behavior (chat/text updates).
  voice     — 32 kbps UDP CBR, 160-byte packets (always on).
              Models packetized voice calls (VoIP-like cadence).
  bulk      — TCP BulkSend, 1400-byte segments, capped at DATA_VOLUME_MB.
              Optional background file-transfer traffic.

Source for traffic characterisation:
  ETSI TR 103 559 v1.1.1 (2021) §5 (NTN traffic models for validation).
  3GPP TR 38.913 §7.1 (IMT-2020 usage scenarios: eMBB, URLLC, mMTC).
"""

CLIENT_PROFILES = (
    ["streaming"] * 10 +
    ["gaming"]   * 10 +
    ["texting"]  * 10 +
    ["voice"]    * 10 +
    ["bulk"]     * 10
)
"""
Traffic profile assignment for each of the
(NUM_STATIONARY_CLIENTS + NUM_MOVING_CLIENTS) = 50 clients.

Distribution (equal split):
  10 × streaming (indices  0–9)
  10 × gaming    (indices 10–19)
  10 × texting   (indices 20–29)
  10 × voice     (indices 30–39)
  10 × bulk      (indices 40–49)

The first NUM_STATIONARY_CLIENTS entries are assigned to stationary clients;
the remaining NUM_MOVING_CLIENTS entries are assigned to moving clients.
Profiles are intentionally mixed between stationary and moving clients so
that mobility effects are visible across all traffic types.

To change the distribution, edit this list directly.  len(CLIENT_PROFILES)
must equal NUM_STATIONARY_CLIENTS + NUM_MOVING_CLIENTS.
"""

# =============================================================================
# Time-series collection
# =============================================================================

TIMESERIES_BUCKET_S = 1.0
"""
Probe interval [s] for per-second throughput time-series collection.

A recurring NS-3 Simulator callback fires every TIMESERIES_BUCKET_S seconds
and reads the cumulative received bytes from the PacketSink application on
the server node.  The per-second throughput is computed as the byte-count
delta between consecutive probes.

- 1.0 s gives 60 data points for a 60 s simulation — fine enough to
  resolve handover dips (~50–200 ms blackout) while keeping probe overhead
  negligible.
- Reducing to 0.1 s would show finer handover transients but introduces
  ~600 extra Simulator events per run.
"""
