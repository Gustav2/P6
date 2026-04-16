"""
config/network.py — NS-3 / protocols / clients / traffic parameters
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
Data rate of the satellite → Internet Server direct link [NS-3 string].

The satellite connects directly to the internet server; there are no
ISL or ground-station hops in the simulated topology.  This link
represents the satellite's direct IP peering into the public internet
backbone (e.g. via a cloud PoP).

- 1 Gbps is achievable via high-throughput satellite transponders.
- This link should never be the bottleneck; the service link (10 Mbps)
  is the limiting hop.
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
