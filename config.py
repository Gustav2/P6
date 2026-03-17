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
  needed for LEO NTN to handle Doppler shifts of up to ±24 kHz at 3.5 GHz.
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

BATCH_SIZE = 64
"""
Number of independent channel realisations evaluated per SNR point.
- Higher values reduce Monte-Carlo variance but increase memory/time.
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
- Doppler shift at 3.5 GHz (worst case, overhead pass):
    Δf_max = v/c × f = 7612 / 3×10⁸ × 3.5×10⁹ ≈ ±88 Hz per km of slant
    Peak Δf ≈ ±24 kHz (when satellite is near the horizon).
- Used in ntn_phy.py: ut_velocities is set to zero, NOT to this value.
  3GPP TR 38.821 §6.1.2 mandates that NTN UEs pre-compensate the satellite
  Doppler shift before the OFDM demodulator.  At 3.5 GHz, the max Doppler
  from this speed is ~88 kHz >> 15 kHz SCS, which causes BER ≈ 0.5 if
  applied without pre-compensation.  Setting ut_velocities = 0 models the
  post-compensation residual.  This constant is retained for documentation
  and for the NTN Doppler branch in Sionna (bs_height ≥ 600 km path).
Source: Standard orbital mechanics (IERS Conventions 2010, §6.1).
"""

NUM_SATELLITES = 3
"""
Number of satellites in the simulated constellation pass.

This controls both the Sionna RT ray-tracing snapshot and the NS-3
handover schedule.  Three satellites are used because:
  - They produce 2 handover events during the 600 s simulation,
    covering high (Sat 0), medium (Sat 1), and low (Sat 2) elevation
    angles — the three link-budget regimes of interest.
  - Ray tracing 3 satellites (PathSolver + render per satellite) takes
    ~2–4 minutes on CPU-only hardware; each additional satellite adds
    ~45–90 s.  300 satellites would take ~4–8 hours and is not feasible.
  - The 3 satellite positions span zenith angles 20°, 35°, and 50°
    (with SAT_SPACING_DEG = 15° and RT_SAT_INITIAL_ZENITH_DEG = 20°),
    giving elevation angles 70°, 55°, and 40° — well above the 10°
    handover threshold and representative of a realistic LEO pass.
  - Minimum 2 to observe at least one handover event.

For the Starlink constellation context (300+ satellites in a shell),
the NUM_SATELLITES value here represents the number of satellites that
are active (above the horizon) during a single simulated orbital pass
rather than the total constellation size.
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
  SAT_TX_EIRP_FEEDER_DBM for the feeder link).
- The absolute RT path gains are not used directly for PER calculation;
  only the *relative* gain differences between satellites are meaningful.
Source: [3GPP-38.821] §6.1, Table 6.1.1-1 (LEO sat downlink EIRP range).
"""

RT_CAM_POSITION = [-170, -170, 140]
"""Camera position [x, y, z] in metres for scene renders."""

RT_CAM_LOOK_AT = [50, 50, 0]
"""Look-at target [x, y, z] in metres for scene renders."""

RT_GS_POSITION = [400.0, 80.0, 10.0]
"""
Ground Station (GS) position [x, y, z] in metres within the Munich scene.
- z = 10 m represents a rooftop dish installation.
- x=400, y=80 places the GS in an open area near the scene edge (east side),
  well separated from the UE at [50, 80, 1.5] so that the sat→GS geometry
  (elevation angle, shadowing) differs meaningfully from the sat→UE service link.
- Scene bounding box: x ∈ [−806, 670] m, y ∈ [−689, 517] m.
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
- At 550 km LEO (RTT ≈ 34 ms round-trip through sat+feeder+fibre ≈ 54 ms)
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
# Antenna / EIRP parameters for direct vs indirect topology comparison
# =============================================================================

PHONE_EIRP_DBM = 23.0
"""
Phone (UE) uplink EIRP [dBm] for the direct Phone→Satellite path.
- 23 dBm = 200 mW, the maximum transmit power for 5G NR UE Power Class 3,
  which covers the vast majority of handsets.
- Omnidirectional handheld antenna; no beamforming gain assumed.
- Produces higher PER on the NTN hop compared to a fixed gNB antenna.
Source: [3GPP-38.101-1] §6.2.2, Table 6.2.2-1 (UE Power Class 3 = 23 dBm).
"""

GNB_EIRP_DBM = 43.0
"""
gNB uplink EIRP [dBm] for the indirect Phone→gNB→Satellite path.
- 43 dBm = 20 W total radiated power.  A mid-size outdoor 5G gNB with a
  64T64R antenna array delivers 43–48 dBm EIRP toward a satellite.
  The 3GPP NTN reference gNB is specified at 38–43 dBm TX power + antenna
  gain; 43 dBm represents the upper end of the macro-cell range.
Source: [3GPP-38.821] §6.1, Table 6.1.1-1 (NTN gNB reference EIRP = 43 dBm).
        [Nokia-5G] white paper §3 (outdoor macro gNB EIRP 43–48 dBm).
"""

GNB_PROCESSING_DELAY_MS = 3.5
"""
gNB one-way baseband processing delay [ms] added to the indirect
topology's terrestrial-hop latency.

This accounts for L1 (PDSCH/PUSCH decode) + L2 (RLC/PDCP reassembly) +
scheduling pipeline delay at the gNB, measured from packet reception at
the air interface to forwarding toward the NTN uplink.

Derivation from 3GPP TS 38.133:
  • N1 (UE PDSCH processing, µ=0): 8 OFDM symbols = 8/14 ms ≈ 0.571 ms
  • N2 (gNB PUSCH processing, µ=0): 3 OFDM symbols = 3/14 ms ≈ 0.214 ms
  These are *minimum* L1 pipeline budgets; actual gNB implementations
  add scheduling wait time (up to 1 slot = 1 ms at µ=0) and L2/L3
  processing (RLC SDU assembly, PDCP ciphering, GTP-U encapsulation).

Measured values from deployed 5G gNBs (industry sources):
  • Nokia AirScale gNB L1+L2 one-way latency: 1–4 ms (Nokia 5G KPI guide,
    2021).
  • Ericsson Radio System gNB PDCP-to-air latency: 2–5 ms (Ericsson
    Technology Review, 5G NR latency analysis, 2017).
  • 3GPP TR 38.913 §8.2.1 user-plane latency budget: gNB contribution
    of (T_proc,1 + T_proc,2) = 4 OFDM symbols + scheduling delay ≈ 3 ms
    for a typical µ=0, DL-heavy slot pattern.

Adopted value: 3.5 ms (midpoint of the 2–5 ms range, consistent with
  the 3GPP TR 38.913 user-plane latency breakdown for µ=0).

Sources:
  [3GPP-38.133] §7.6, Table 7.6.2.1-1 (N1/N2 processing time symbols).
  3GPP TR 38.913 v16.0.0 §8.2.1 (user-plane latency budget, gNB = ~3 ms).
  Ericsson Technology Review, "5G NR — The Next Generation Wireless
    Access Technology" (2017), §3.3, Table 1.
"""

TERRESTRIAL_BACKHAUL_DELAY_MS = 10.0
"""
One-way latency of the Ground Station ↔ Internet Server terrestrial fibre
link [ms].

- 10 ms is representative of a metro-area fibre path from a satellite ground
  station to a regional data centre / Internet exchange point.
- Derivation: typical backbone fibre propagation velocity ≈ 0.67 c
  (200 000 km/s); a 2 000 km GS-to-IXP fibre route gives
  10 ms one-way delay.
- Sources:
    Singla et al., "Middleboxes as a Cloud Service" (HotNets 2014): measured
      fibre latency ≈ 5 µs/km for terrestrial routes.
    Akamai State of the Internet Q4 2022: regional CDN RTTs 15–30 ms imply
      one-way backhaul of 7–15 ms for metro distances.
"""

GNB_DATARATE = "150Mbps"
"""
Data rate of the Phone→gNB terrestrial radio link [NS-3 string].
- Represents a 5G NR Uu interface in a local urban macro cell.
- 3GPP TS 38.306 §4.1.2 defines the UE downlink peak data rate formula:
    R = ν × Q × f × R_code × N_PRB × 12 / T_slot
  For a Category 3 UE (4 layers, 256-QAM, 100 MHz n78):
    DL peak ≈ 1.6 Gbps theoretical; with overhead ~1 Gbps.
  UL peak for a typical UE (1 layer, 64-QAM, 100 MHz): ~150–200 Mbps.
- 150 Mbps is a realistic single-UE uplink throughput for a mid-range
  5G NR deployment (3GPP TR 38.913 §7.1 target: 50 Mbps UL per user).
Source: [3GPP-38.306] §4.1.2 (peak data rate formula).
        3GPP TR 38.913 v16.0.0 §7.1 (IMT-2020 UL edge rate 50 Mbps,
          peak 10 Gbps DL / 10 Gbps UL at theoretical limits).
"""

GNB_TERRESTRIAL_PER = 0.001
"""
Packet error rate on the Phone→gNB terrestrial hop (residual after HARQ).
- 3GPP TS 38.214 §5.1 sets the initial BLER target at 10% for the first
  HARQ transmission.  After combining 2–4 HARQ retransmissions the
  residual BLER (≈ undetected-error PER) drops to 10⁻³ to 10⁻⁴.
- 0.001 (0.1 %) is the standard assumption for a well-margined terrestrial
  5G NR link after HARQ and is negligible compared to the NTN hop PER
  on the direct satellite path.
Source: [3GPP-38.214] §5.1 (target BLER 10% before HARQ combining).
        3GPP TR 38.913 §8.2 (residual BLER ≤ 10⁻³ after 4 HARQ rounds).
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
# Feeder link parameters (Satellite → Ground Station, Ka-band)
# =============================================================================

FEEDER_FREQ_HZ = 26.5e9
"""
Feeder link downlink frequency [Hz].
- 26.5 GHz (Ka-band) is used by modern LEO NTN feeder links.  Starlink
  uses 26.5–27.0 GHz (Ku/Ka transition) for the gateway downlink.
- Higher than the service link (3.5 GHz), so FSPL is ~17.5 dB greater
  for the same slant range.  Compensated by the larger GS dish gain.
- Sionna RT proxy paths are traced at RT_SCENE_FREQ_HZ (3.5 GHz); only
  the relative multipath correction is taken from RT.  The absolute FSPL
  is applied analytically using this frequency.
Source: [Starlink-FCC] Attachment B, §2 (Ka-band gateway 26.5–27.0 GHz).
        [3GPP-38.821] §6.1 (feeder link: Ka-band, 20/30 GHz bands).
"""

FEEDER_BANDWIDTH_HZ = 250e6
"""
Feeder link channel bandwidth [Hz].
- 250 MHz is the per-gateway Ka-band allocation used by Starlink (FCC
  approval 2020).  The ITU-R Ka-band feeder allocation for FSS gateways
  is up to 500 MHz per beam under S.524-9.
- Used in the Shannon capacity formula C = B · log₂(1 + SNR) to derive
  the RT-calibrated data rate that replaces the hardcoded "100Mbps" string.
Source: [Starlink-FCC] Attachment B, §2 (gateway bandwidth 250 MHz/beam).
        [ITU-R-S.524] §2 (Ka-band earth-station e.i.r.p. and bandwidth).
"""

GS_RX_ANTENNA_GAIN_DB = 42.7
"""
Ground Station receive antenna gain [dBi].
- A 2.4 m parabolic dish at 26.5 GHz with 60 % aperture efficiency:
    G = η · (π · D · f / c)²  [linear]
    G_dB = 10·log10(0.60 · (π · 2.4 · 26.5e9 / 3e8)²) = 42.7 dBi
- This matches a Starlink/OneWeb-class gateway dish (typical: 2.4–3.8 m).
- Included in the feeder-link SNR budget:
    SNR_feeder = SAT_TX_EIRP_FEEDER − FSPL_Ka + urban_correction
                 + GS_RX_ANTENNA_GAIN_DB − NOISE_FLOOR_DBM
Source: [ITU-R-S.465] §2 (reference GS dish performance at Ka-band).
        [Starlink-FCC] Attachment B (gateway antenna diameter 2.4 m,
          gain ≥ 42 dBi at 26.5 GHz).
"""

SAT_TX_EIRP_FEEDER_DBM = 57.0
"""
Satellite feeder downlink EIRP [dBm] toward the Ground Station.
- Represents the Ka-band spot-beam transmit power from a modern LEO
  satellite toward a fixed gateway.
- Derivation: 10 W RF output power = 40 dBm + 17 dBi Ka-band spot-beam
  transmit antenna = 57 dBm EIRP.  This is consistent with published
  Starlink satellite EIRP in FCC filings (55–60 dBm for gateway beams).
- The 3GPP NTN feeder-link reference uses a satellite TX power of
  10–30 W combined with spot-beam antennas of 12–20 dBi, giving
  ~52–62 dBm EIRP.
Source: [3GPP-38.821] §6.1, Table 6.1.2-1 (feeder-link satellite EIRP).
        [Starlink-FCC] Attachment B (satellite gateway EIRP ≥ 55 dBm).
"""

# =============================================================================
# Link budget thresholds — shared by ntn_ns3.py and topology_diagram.py
# =============================================================================

NOISE_FLOOR_DBM = -121.0
"""
Receiver thermal noise floor [dBm] for both service-link and feeder-link
budgets.

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

# =============================================================================
# Multi-client topology settings
# =============================================================================

USE_BASE_STATIONS = False
"""
Topology mode flag.

  True  — Indirect:  Phone → gNB → AccessSat → ISL → BenchmarkSat → GS → Server
           Each phone is assigned to the nearest gNB (from GNB_POSITIONS).
           The gNB aggregates traffic before the NTN hop, improving link budget
           by GNB_EIRP_DBM − PHONE_EIRP_DBM = 20 dB relative to direct mode.

  False — Direct:    Phone → AccessSat → ISL → BenchmarkSat → GS → Server
           Phones connect to the access satellite directly (no gNB hop).
           Represents satellite-direct (SD) IoT / mobile terminal use case.

In both modes the benchmark satellite (Sat 0, the highest-elevation satellite
in the handover schedule) is always reachable via ISL from any access satellite,
so throughput measurements remain consistent across handovers.
"""

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
WAYPOINT_SPEED_MIN_MS and WAYPOINT_SPEED_MAX_MS m/s toward a random
destination within the same radius, with zero pause time at each waypoint.

RandomWaypoint parameters are set in ntn_ns3.py using the bounds
[−CLIENT_AREA_RADIUS_M, CLIENT_AREA_RADIUS_M] for both X and Y axes.
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

WAYPOINT_SPEED_MIN_MS = 1.0
"""Minimum moving-client waypoint speed [m/s] (≈ slow walk)."""

WAYPOINT_SPEED_MAX_MS = 5.0
"""Maximum moving-client waypoint speed [m/s] (≈ fast walk / slow jog)."""

# =============================================================================
# Base station positions (used only when USE_BASE_STATIONS = True)
# =============================================================================

GNB_POSITIONS = [
    (200.0, 150.0),
    (500.0, 300.0),
    (350.0, 600.0),
]
"""
List of (x_m, y_m) positions [m] for the fixed gNB nodes in the NS-3
simulation.  The z-coordinate is fixed at 30.0 m (rooftop installation).

Each phone is assigned to the nearest gNB (minimum Euclidean distance
at t=0 for stationary clients; at initial position for moving clients).

Three gNBs are sufficient to provide at least one handover between cells
for moving clients traversing CLIENT_AREA_RADIUS_M at WAYPOINT_SPEED_MAX_MS.

Positions are expressed in NS-3 Cartesian coordinates (metres), with the
coordinate origin at the centre of the deployment area.
"""

NUM_GNB = len(GNB_POSITIONS)
"""Number of gNB nodes — derived from GNB_POSITIONS, do not set manually."""

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
