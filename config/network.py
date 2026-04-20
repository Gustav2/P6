"""
config/network.py — NS-3 / protocols / clients / traffic parameters
"""

import os

# =============================================================================
# Reproducibility controls
# =============================================================================

SIM_SEED = int(os.environ.get("SIM_SEED", "42"))
"""Global deterministic seed used by Python/NumPy/NS-3 RNGs."""

SIM_RUN = int(os.environ.get("SIM_RUN", "1"))
"""NS-3 run index (RngRun) for controlled multi-run sweeps."""

PY_RANDOM_SEED = SIM_SEED
"""Seed for Python's random module in simulation helpers."""

# =============================================================================
# Scenario selector
# =============================================================================
#
# The simulation supports two calibrated scenarios, selected via the
# ``SIM_SCENARIO`` environment variable:
#
#   "contended" (default) — 50 clients in one beam, mixed traffic profiles.
#                           The realistic urban stress test for the thesis.
#   "empirical"           — 3 bulk clients, 100 Mbps beam (Sander IMC 2022
#                           single-user Ka-band envelope).  Used to match
#                           published measurement conditions so the
#                           empirical-validation plots can compare apples
#                           to apples.
#
# Nothing else is gated on this flag — the PHY, RT, channel, and handover
# models run identically in both modes.  Only client count, traffic mix,
# and service-link capacity differ.
SIM_SCENARIO = os.environ.get("SIM_SCENARIO", "contended").strip().lower()
if SIM_SCENARIO not in ("contended", "empirical"):
    raise ValueError(
        f"SIM_SCENARIO={SIM_SCENARIO!r} not recognised; "
        "expected 'contended' or 'empirical'"
    )

# =============================================================================
# NS-3 network simulation parameters
# =============================================================================

SIM_DURATION_S = 300.0
"""
Total NS-3 simulation duration [seconds].
- 300 s (5 min) covers a substantial portion of one LEO overhead pass
  (~8 min total at 550 km / 7612 m/s) and includes multiple handover
  events with NUM_SATELLITES = 8 and SAT_SPACING_DEG = 15°.
- At ω ≈ 0.063°/s the satellite advances ~18.9° over 300 s, giving
  meaningful orbital geometry variation across RT snapshots.
- NS-3 runtime scales roughly linearly with SIM_DURATION_S × NUM_CLIENTS;
  300 s is a practical upper bound for CPU-only runs.
"""

RT_SNAPSHOT_INTERVAL_S = 15.0
"""
Time interval between successive Sionna RT snapshots [seconds].

At 300 s simulation duration this produces 21 snapshots
(t = 0, 15, 30, …, 300 s), each advancing the satellite constellation
by ≈ 0.95° along the orbit (ω ≈ 0.063°/s at 550 km).  Per-slot channel
statistics in NS-3 are linearly interpolated between adjacent snapshots.

- 15 s / 0.95° per interval gives finer temporal resolution than the
  previous 30 s / 1.9° setting, better capturing how K-factor and delay
  spread vary as each satellite's elevation angle changes over the pass.
- Interpolation error across 0.95° intervals is negligible compared to
  other model uncertainties (coplanar geometry, specular-only RT).
- Each additional snapshot adds one PathSolver call; scene is loaded once
  and reused so the overhead is PathSolver time only (not scene load).
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

TRANSPORT_COMPARE_MODE = os.environ.get(
    "TRANSPORT_COMPARE_MODE", "matched_offered_load"
).strip().lower()
"""
Transport comparison mode.

- "matched_offered_load" (default): all protocols use equivalent offered-load
  traffic generators per profile (rate-controlled OnOff), so differences are
  attributable to transport behavior rather than load mismatch.
- "legacy_mixed_apps": preserve historical app selection (UDP profiles via
  OnOff, bulk via TCP BulkSend in TCP runs).
"""

if TRANSPORT_COMPARE_MODE not in ("matched_offered_load", "legacy_mixed_apps"):
    raise ValueError(
        f"TRANSPORT_COMPARE_MODE={TRANSPORT_COMPARE_MODE!r} not recognised; "
        "expected 'matched_offered_load' or 'legacy_mixed_apps'"
    )

# Matched offered-load rates used when TRANSPORT_COMPARE_MODE is enabled.
# This table is intentionally explicit so protocol parity is reproducible.
MATCHED_PROFILE_DATA_RATES = {
    "streaming": "1.5Mbps",
    "gaming": "120kbps",
    "texting": "30kbps",
    "voice": "32kbps",
    "bulk": "1.5Mbps",
}
"""Per-profile offered-load rates for matched-transport comparisons."""

QUIC_IS_ANALYTICAL = True
"""
Marks QUIC results as analytical post-processed estimates over a TCP BBR
baseline (not packet-level QUIC stack simulation).
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

SAT_RX_ANTENNA_GAIN_DB = 34.0
"""
Satellite receive antenna gain [dBi] applied to the uplink link budget.
- Models the satellite's phased-array / multi-beam receive aperture at
  2 GHz (S-band service link).
- The 3GPP TR 38.821 minimum reference is 30 dBi; operational LEO
  satellites use larger phased arrays:
    - ~256-element array at 2 GHz (λ = 15 cm, 0.5λ spacing, ~0.6 m²
      aperture): peak gain ≈ 10·log10(256·0.65) ≈ 32 dBi per element group.
    - Starlink-class dishes use shaped-beam phased arrays achieving
      34–38 dBi per spot beam in published FCC technical filings.
  34 dBi is a conservative but realistic operational value.
Source: [3GPP-38.821] §6.1, Table 6.1.1-1 (minimum reference 30 dBi).
         SpaceX FCC Technical Appendix (Starlink V2 Mini service link).
"""

# =============================================================================
# Service-link (phone ↔ satellite) shared-beam parameters
# =============================================================================

SERVICE_LINK_RATE_MBPS = 20.0
"""
Aggregate shared service-link (phone → satellite) beam capacity [Mbps].

- All UEs in the simulated cell share this capacity via a CSMA bus in NS-3,
  mimicking the shared radio resource of a single LEO spot beam.
- 20 Mbps is the realistic per-beam uplink capacity for one S-band (2 GHz)
  NR-NTN spot beam with 20 MHz bandwidth at 16QAM r=0.5 (per 3GPP TR 38.821
  §6.1.1 link budget tables).  It matches Starlink DTC FCC filings for
  S-band direct-to-phone spot beams (5–40 Mbps per beam, 20 Mbps typical).
- This is the realistic bottleneck for the 50-UE urban deployment — the
  beam is shared, not per-UE, so aggregate throughput cannot exceed this.
- Lower this to 10 Mbps to model worst-case QPSK r=0.5 cell-edge;
  raise to 40 Mbps for best-case zenith 64QAM.
Source: [3GPP-38.821] §6.1.1, Table 6.1.1-1; SpaceX FCC DTC filings.
"""

# =============================================================================
# Satellite → Internet Server (backhaul) link parameters
# =============================================================================

SAT_SERVER_DATARATE = "1Gbps"
"""
Data rate of the satellite → Internet Server backhaul link [NS-3 string].

- 1 Gbps represents a high-capacity feeder / trunk link that is never the
  bottleneck; the service-link beam is the limiting hop.
- Applies to the sat → server point-to-point channel only.
"""

BACKHAUL_DELAY_MS = 15.0
"""
Satellite → Internet Server one-way backhaul delay [ms].

- Represents the gateway (ground-station) hop plus Internet transit from the
  gateway PoP to the application server.
- 15 ms is consistent with measured Starlink ground-segment latency:
    * Sat → gateway feeder link (Ka/Ku): ~2–3 ms one-way
    * Gateway PoP → major Internet backbone: ~5–10 ms (continental)
    * ISP edge → application server: ~3–5 ms
  Total service-link (≈5 ms) + backhaul (15 ms) + return path gives
  ~40–50 ms RTT end-to-end, matching Sander et al. IMC 2022 measurements.
Source: Sander et al. "Measuring the Performance of Satellite Broadband"
        IMC 2022; Starlink FCC ex parte filings on gateway architecture.
"""

# =============================================================================
# Link budget thresholds — shared by ntn_ns3.py and topology_diagram.py
# =============================================================================

NOISE_FLOOR_DBM = -118.0
"""
Receiver thermal noise floor [dBm] for the service-link budget.

Derivation (Johnson–Nyquist thermal noise, per NR subcarrier):
  N₀ = k · T · B
  k  = 1.380649×10⁻²³ J/K  (Boltzmann constant)
  T  = 290 K                (ITU-R reference noise temperature, [ITU-R-S.465])
  B  = Δf_sc = 30 kHz       (one NR subcarrier, numerology µ=1; this is the
                              natural noise bandwidth for per-subcarrier SNR
                              as used throughout the link budget and Sionna
                              ResourceGrid, where SNR is defined per subcarrier)

  kTB(30 kHz) = 10·log10(1.38e-23 × 290 × 30e3) + 30  [dBm]
              = 10·log10(1.201e-16) + 30
              = −159.2 + 30 = −129.2 dBm/subcarrier

  (Cross-check: kTB = −174 dBm/Hz + 10·log10(30e3)
                     = −174 + 44.8 = −129.2 dBm ✓)

  + Receiver noise figure (NF) — composite system budget:
    - UE/phone S-band LNA: NF ≈ 6–7 dB (3GPP TS 38.101-1 §7.3 reference
      sensitivity assumes NF = 7 dB for UE).
    - Feed/connector losses at phone: ~1–2 dB.
    - ADC quantisation + baseband processing: ~0.5–1 dB.
    - Adopted composite NF = 11.2 dB (consistent with 3GPP TR 38.821 §6.1.1
      UL link budget, Table 6.1.1-1, which assumes UE NF = 10 dB and
      accounts for 1–2 dB of additional implementation loss).

  N_floor = kTB(30 kHz) + NF = −129.2 + 11.2 = −118 dBm/subcarrier

  Note: switching from µ=0 (15 kHz) to µ=1 (30 kHz) raises the per-
  subcarrier noise floor by 3 dB (double the bandwidth), but the 4.9 dB
  FSPL saving from 3.5 GHz → 2.0 GHz gives a net +1.9 dB link margin
  improvement at 550 km nadir.

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
# ITU-R P.618 atmospheric / rain fade parameters
# =============================================================================

ATMO_GASEOUS_DB = 0.3
"""
Clear-sky gaseous absorption [dB] at 2 GHz per ITU-R P.676-12.
- Combined O₂ + H₂O absorption at S-band is dominated by water vapour;
  at ~7.5 g/m³ reference humidity and 90° elevation this sums to ~0.3 dB.
- Scales with 1/sin(elev) down to the 10° horizon floor (already accounted
  for inside _itu_p618_atm_db).
Source: ITU-R P.676-12 §2.2 (specific attenuation × slant-path length).
"""

RAIN_RATE_MM_H = 12.0
"""
Rain rate exceeded 0.01% of the year [mm/h] for temperate climate zone.
- ITU-R P.837-7 Zone K nominal value (covers most of northern Europe).
- Drives the ITU-R P.618-13 rain-fade computation for link availability.
- Raise to 42 mm/h (Zone P, tropical maritime) or lower to 8 mm/h (Zone E,
  temperate-dry) to study different climate regimes.
Source: ITU-R P.837-7 Annex 1; ITU-R P.618-13 §2.2.1.
"""

RAIN_AVAILABILITY_PCT = 99.0
"""
Target link availability [%] for the rain-fade budget.
- 99.0% → accepts rain fade exceeded 1% of the year (worst-case 3.65 days).
- 99.9% → more stringent (exceeded 0.1% of the year) — fade margin grows ~2×.
- At S-band (2 GHz) rain is not a dominant impairment; this value is retained
  for methodological completeness rather than real availability impact.
"""

ATMO_SCINTILLATION_DB = 0.2
"""
Tropospheric scintillation standard deviation [dB] at 2 GHz.
- ITU-R P.618-13 §2.4.1 gives σ_scint proportional to f^(7/12) / sin(elev)^1.2.
- At 2 GHz, clear-sky σ_scint ≤ 0.2 dB even at 10° elevation — negligible
  compared to the 4 dB shadow fading, but included for completeness.
Source: ITU-R P.618-13 §2.4.1 (scintillation model).
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

DATA_VOLUME_MB = 100.0
"""
Maximum data volume sent per TCP BulkSend flow [MB].

NS-3 BulkSend.MaxBytes is set to int(DATA_VOLUME_MB × 10⁶) bytes so that
each flow terminates after transferring this fixed file size rather than
saturating the link for the full SIM_DURATION_S.

- 100 MB keeps BulkSend flows alive for the full 300 s simulation at NTN
  rates (≤20 Mbps), so the TCP congestion controller reaches steady state
  and the fairness index is computed over a complete flow. At 20 Mbps a
  10 MB transfer finishes in ~4 s, leaving the congestion controller in
  slow-start for most of the window — 100 MB prevents this early cutoff.
- Set to 0 to disable the cap (unlimited BulkSend, saturating sender).

UDP OnOff flows are not capped by this value (they use APP_DATA_RATE).
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

HANDOVER_INTERRUPTION_MS_MAX = 300.0
"""
Maximum total link-layer interruption time per handover event [ms].

300 ms is the upper bound of the 3GPP NTN CHO interruption range
(50–300 ms) for conventional LEO handover without pre-positioning.
It covers worst-case BFD timer expiry + congested RACH + slow RRC path
switch under high orbital Doppler and multipath uncertainty.
Source: 3GPP TR 38.821 §6.2.2 (NTN realistic range 50–300 ms).
        3GPP TS 38.321 §5.17 (BFD max 200 ms; RACH + CHO add remainder).
"""

HANDOVER_PHASE_JITTER_MS = 5.0
"""
Uniform random jitter magnitude [ms] applied independently to BFD, RACH,
and CHO phase durations when building handover interruption events.
"""

FADE_MEAN_DURATION_MS = 80.0
"""
Mean duration of a shadow-fading burst on the NTN service link [ms].

Shadow fading in urban NTN has a coherence time set by the satellite's
angular velocity across building edges. At 550 km altitude and 7612 m/s,
the satellite subtends ~1.4 m/s apparent lateral motion at street level,
giving a fading coherence length L_c ≈ 10–100 m (1–10 dB fade depth).

The speed-dependent mean fade duration is:
    T_fade(v_UE) = L_c / v_rel,  where v_rel ≈ v_UE (satellite Doppler
    dominates only for the frequency component; the shadow geometry is
    determined by UE-to-building relative motion at street level).

At pedestrian speed (~1.4 m/s):  T_fade ≈ 10–70 m / 1.4 m/s ≈ 70 ms  ✓
At vehicular speed  (~15 m/s):   T_fade ≈ 10–70 m / 15   m/s ≈  5 ms
At highway speed    (~30 m/s):   T_fade ≈ 10–70 m / 30   m/s ≈  2 ms

80 ms is therefore appropriate only for pedestrian UEs. Vehicular UEs in
this simulation (RandomWaypoint, speeds up to ~15 m/s) will experience
shadow-fading bursts 8–16× shorter than modelled here. This overestimates
burst duration and may slightly overstate protocol benefit from burst
recovery (QUIC vs TCP). Documented as thesis caveat N1-a.

Reference: 3GPP TR 38.811 §6.7.2 urban S-band shadow-fading statistics.
Used in the burst-error state machine in sim/ns3.py (N1 fix).
"""

# =============================================================================
# Traffic profiles and client assignment
# =============================================================================

TRAFFIC_PROFILES = {
    "streaming": {
        "app_type":   "udp_cbr",
        "data_rate":  "1.5Mbps",
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
  streaming — 1.5 Mbps UDP CBR, 1316-byte packets (always on).
              Models 480p adaptive video streaming — realistic for S-band DTC
              where per-UE bandwidth is constrained (ETSI TR 103 559 §5).
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

TRAFFIC_PROFILE_COUNTS = {
    "streaming": 10,
    "gaming":    10,
    "texting":   10,
    "voice":     10,
    "bulk":      10,
}
"""
Per-profile client counts.  Must sum to NUM_STATIONARY_CLIENTS + NUM_MOVING_CLIENTS.

To change traffic mix (e.g. heavy streaming + light bulk), just edit the
counts — CLIENT_PROFILES below is derived automatically.  Add a new profile
by adding an entry to TRAFFIC_PROFILES *and* to this dict.
"""

CLIENT_PROFILES = [
    name
    for name, count in TRAFFIC_PROFILE_COUNTS.items()
    for _ in range(count)
]
"""
Traffic profile assignment per client, derived from TRAFFIC_PROFILE_COUNTS.

The first NUM_STATIONARY_CLIENTS entries are assigned to stationary clients;
the remaining NUM_MOVING_CLIENTS entries are assigned to moving clients.
len(CLIENT_PROFILES) must equal NUM_STATIONARY_CLIENTS + NUM_MOVING_CLIENTS.
"""

_expected_total = NUM_STATIONARY_CLIENTS + NUM_MOVING_CLIENTS
if len(CLIENT_PROFILES) != _expected_total:
    raise ValueError(
        f"TRAFFIC_PROFILE_COUNTS sums to {len(CLIENT_PROFILES)} clients, "
        f"expected {_expected_total} (NUM_STATIONARY_CLIENTS + NUM_MOVING_CLIENTS)."
    )


# =============================================================================
# Empirical-scenario overrides
# =============================================================================
# When SIM_SCENARIO="empirical", replace the contended-urban deployment
# with a lightly-loaded cell matching the measurement conditions of
# Sander et al. IMC 2022 (single-user Ka-band envelope, ~100 Mbps beam).
# We reuse the constant names above; only these values change.
if SIM_SCENARIO == "empirical":
    NUM_STATIONARY_CLIENTS        = 2
    NUM_MOVING_CLIENTS            = 1
    NUM_PEDESTRIAN_MOVING_CLIENTS = 1
    NUM_VEHICULAR_MOVING_CLIENTS  = 0
    SERVICE_LINK_RATE_MBPS        = 100.0
    TRAFFIC_PROFILE_COUNTS        = {"bulk": 3}
    CLIENT_PROFILES               = ["bulk", "bulk", "bulk"]
    # No need to re-validate the sum — NUM_STATIONARY_CLIENTS +
    # NUM_MOVING_CLIENTS = 3 and len(CLIENT_PROFILES) = 3 by construction.

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
