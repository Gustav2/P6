"""
ntn_ns3.py — Part 2: NS-3 packet-level 5G-NTN network simulation
=================================================================
Simulates the end-to-end NTN network including:

  * A moving LEO satellite constellation — satellites pass overhead and
    the UE hands over to the next satellite when the serving one drops
    below SAT_HANDOVER_ELEVATION_DEG.
  * RT-informed channel model — the packet error rate on the service
    link is derived from the mean path gain computed by Sionna RT in
    rt_sim.py, rather than from a purely analytic FSPL formula.
  * Multi-protocol comparison — the function run_ns3_both_topologies()
    runs the full simulation once per entry in config.PROTOCOLS × 2
    topologies (direct / indirect) and returns all results for plotting.
  * QUIC emulation — QUIC is emulated on top of NS-3 UDP with RFC 9000
    corrections applied analytically after FlowMonitor collection.
  * Full beam management handover — a 3-phase link interruption gap
    models beam failure detection → RACH → conditional handover
    per 3GPP TS 38.300 §10.1.2.3.
  * Mixed traffic profiles — each client is assigned a traffic profile
    (streaming / gaming / texting / voice / bulk) so that heterogeneous load is
    accurately modelled across all protocol comparison runs.
  * Per-second time-series — a PacketSink probe fires every
    TIMESERIES_BUCKET_S seconds to record true per-second throughput.
  * Jain's fairness index — computed across per-flow throughput values
    after FlowMonitor collection.

Topologies
----------

  Direct (4 nodes):
    Phone (UE)
       │  5G-NR NTN service link  (550 km LEO, RT-calibrated PER, PHONE_EIRP)
       ▼
    Satellite [moving]     ← handover when elevation < SAT_HANDOVER_ELEVATION_DEG
       │  ISL  (1 Gbps, 5 ms)
       ▼
    Benchmark Satellite
       │  Direct IP link  (1 Gbps, 10 ms)
       ▼
    Internet Server

  Indirect (5 nodes):
    Phone (UE)
       │  Terrestrial radio  (GNB_DATARATE, 1 ms, GNB_TERRESTRIAL_PER)
       ▼
    gNB (fixed antenna)
       │  5G-NR NTN service link  (550 km LEO, RT-calibrated PER, GNB_EIRP)
       ▼
    Satellite [moving]
       │  ISL  (1 Gbps, 5 ms)
       ▼
    Benchmark Satellite
       │  Direct IP link  (1 Gbps, 10 ms)
       ▼
    Internet Server

Beam management handover (3GPP TS 38.300 §10.1.2.3)
----------------------------------------------------
Each handover consists of three sequential phases:

  Phase 1 — Beam Failure Detection (BEAM_FAILURE_DETECTION_MS):
    The UE detects that the serving beam has dropped below the handover
    threshold.  During this window the link is unreliable (PER → 1.0).

  Phase 2 — Random Access (RANDOM_ACCESS_DELAY_MS):
    The UE performs PRACH on the target satellite beam.

  Phase 3 — Conditional Handover Preparation (CONDITIONAL_HO_PREP_MS):
    RRC reconfiguration and path switch complete.  The link recovers
    with the new satellite's PER.

Total interruption gap is drawn uniformly from
[HANDOVER_INTERRUPTION_MS_MIN, HANDOVER_INTERRUPTION_MS_MAX].
The gap applies to the NTN hop in both direct and indirect topologies.

QUIC emulation (RFC 9000 / RFC 9002)
-------------------------------------
QUIC has no native NS-3 module; it is emulated over UDP with mechanism-by-
mechanism post-FlowMonitor corrections derived from first principles:

  1. 1-RTT handshake saving:    QUIC needs 1 RTT vs TCP's 2 RTT to start
                                 sending data.  Extra throughput credit =
                                 (mean_RTT_s × link_rate_bps) × 8 / active_s.

  2. PTO vs RTO (no cwnd→1):   QUIC's Probe Timeout fires probe packets
                                 WITHOUT collapsing cwnd to 1 MSS.  At
                                 high PER, TCP RTO events collapse cwnd
                                 repeatedly; QUIC recovers faster.
                                 Credit = cwnd_floor × MSS × rto_rate × 8 / 1e3

  3. Unlimited ACK ranges:      QUIC ACK frames support unlimited gap/range
                                 pairs vs TCP SACK's 3 blocks.  At PER > 0.5
                                 the loss map requires many ACK round trips
                                 for TCP; QUIC resolves it in one.  Applied
                                 at high-PER slots only.

  4. Post-handover recovery:    QUIC PATH_CHALLENGE/RESPONSE costs 1 RTT and
                                 preserves ssthresh; TCP breaks the connection
                                 entirely.  The post-handover throughput dip
                                 is reduced vs TCP BBR.  Beam interruption gap
                                 is shared by both protocols; only recovery
                                 RTTs differ.

  5. HoL blocking:              NOT applied — zero benefit for single bulk
                                 stream (only matters for multi-stream HTTP/3).

The QUIC baseline is TCP BBR (BBR is the congestion controller used by
real QUIC implementations such as Chromium/quiche and lsquic for satellite
links).  Corrections are additive on top of the BBR aggregate result.

Dependencies
------------
  NS-3 with Python bindings (cppyy-based, 3.37+).
  See README.md for build instructions.
"""

import math
import random as _random_mod
import numpy as np
from ns import ns   # NS-3 cppyy-based Python bindings

# ---------------------------------------------------------------------------
# C++ helper: wrap a no-arg Python callable as an NS-3 EventImpl*
# This is required because the cppyy bindings cannot dispatch plain Python
# functions to ns3::Simulator::Schedule directly.  The pattern comes from
# the official NS-3 Python example tcp-bbr-example.py.
# ---------------------------------------------------------------------------
ns.cppyy.cppdef(r"""
#include "ns3/simulator.h"
using namespace ns3;
EventImpl* pythonMakeEvent(void (*f)()) { return MakeEvent(f); }
""")

from config import (
    CARRIER_FREQ_HZ,
    SAT_HEIGHT_M,
    SAT_ORBITAL_VELOCITY_MS,
    SAT_HANDOVER_ELEVATION_DEG,
    SIM_DURATION_S,
    APP_DATA_RATE,
    PACKET_SIZE_BYTES,
    NUM_PARALLEL_FLOWS,
    TCP_SNDRCV_BUF_BYTES,
    TCP_SACK_ENABLED,
    TCP_TIMESTAMPS,
    PROTOCOLS,
    PHONE_EIRP_DBM,
    GNB_EIRP_DBM,
    GNB_PROCESSING_DELAY_MS,
    TERRESTRIAL_BACKHAUL_DELAY_MS,
    GNB_DATARATE,
    GNB_TERRESTRIAL_PER,
    SAT_RX_ANTENNA_GAIN_DB,
    NOISE_FLOOR_DBM,
    SNR_THRESH_DB,
    SIGMOID_SLOPE,
    RT_GAIN_P10_BLEND,
    # Multi-client topology
    USE_BASE_STATIONS,
    NUM_STATIONARY_CLIENTS,
    NUM_MOVING_CLIENTS,
    CLIENT_AREA_RADIUS_M,
    NUM_PEDESTRIAN_MOVING_CLIENTS,
    NUM_VEHICULAR_MOVING_CLIENTS,
    PEDESTRIAN_SPEED_MIN_MS,
    PEDESTRIAN_SPEED_MAX_MS,
    VEHICULAR_SPEED_MIN_MS,
    VEHICULAR_SPEED_MAX_MS,
    GNB_POSITIONS,
    NUM_GNB,
    DATA_VOLUME_MB,
    ISL_DATARATE,
    ISL_DELAY_MS,
    ISL_PER,
    SAT_SERVER_DATARATE,
    # Beam management
    HANDOVER_INTERRUPTION_MS_MIN,
    HANDOVER_INTERRUPTION_MS_MAX,
    # Traffic profiles
    TRAFFIC_PROFILES,
    CLIENT_PROFILES,
    # Time-series
    TIMESERIES_BUCKET_S,
)


# =============================================================================
# Geometry and link-budget helpers
# =============================================================================

def _slant_range_m(height_m: float, elev_deg: float) -> float:
    """
    One-way slant range from UE to satellite [m] using spherical Earth
    geometry.

    Parameters
    ----------
    height_m : float  Satellite orbital altitude [m].
    elev_deg : float  UE-to-satellite elevation angle [degrees].
    """
    RE = 6_371_000.0
    e  = math.radians(elev_deg)
    return (math.sqrt((RE + height_m) ** 2 - (RE * math.cos(e)) ** 2)
            - RE * math.sin(e))


def _one_way_delay_ms(height_m: float, elev_deg: float) -> float:
    """One-way propagation delay [ms] along the slant range."""
    return _slant_range_m(height_m, elev_deg) / 3e8 * 1e3


def _fspl_db(height_m: float, elev_deg: float,
             fc_hz: float = CARRIER_FREQ_HZ) -> float:
    """
    Free-space path loss [dB] at carrier frequency ``fc_hz``.

    FSPL = 20·log10(4π·d·f / c)
    """
    d = _slant_range_m(height_m, elev_deg)
    return 20.0 * math.log10(4.0 * math.pi * d * fc_hz / 3e8)


def _rt_calibrated_per(fspl_db: float, rt_mean_gain_db: float,
                        rt_gain_p10_db: float = None,
                         rt_ref_gain_db: float = None,
                         tx_eirp_dbm: float = None,
                         snr_thresh_db: float = None,
                        sigmoid_slope: float = None) -> float:
    """
    Estimate packet error rate using a sigmoid link-budget model
    calibrated by the RT-derived mean path gain.

    The Sionna RT proxy TX is placed only RT_SAT_SCENE_HEIGHT_M (~300 m)
    above the scene, so the absolute mean_path_gain_db value reflects both
    the short proxy free-space loss and urban multipath.  The *relative*
    gain difference between satellites (due to elevation-dependent shadowing
    and multipath) is physically meaningful; the absolute value is not
    directly transferable to the real 550 km slant range.

    Link budget:
        SNR [dB] = Tx_EIRP − FSPL_600km + urban_correction − Noise_floor

    where urban_correction is derived by normalising the RT gain against the
    best (highest-elevation) satellite so that the reference satellite
    contributes 0 dB correction and lower-elevation satellites are penalised
    by their relative RT gain deficit.

    Parameters
    ----------
    fspl_db         : float  Free-space path loss over real 550 km slant [dB].
    rt_mean_gain_db : float  Mean |h| gain from Sionna RT for this satellite [dB].
    rt_ref_gain_db  : float  RT gain of the reference (best) satellite [dB].
                             When None, no urban correction is applied.
    tx_eirp_dbm     : float  Transmitter EIRP [dBm].  When None, defaults to
                             PHONE_EIRP_DBM (23 dBm) for backward compatibility.
                             Use GNB_EIRP_DBM (43 dBm) for the indirect topology.
    snr_thresh_db   : float  Sigmoid threshold [dB].  When None, uses
                             SNR_THRESH_DB from config (initial estimate).
                             Pass the value fitted to the Sionna LDPC BER curve.
    sigmoid_slope   : float  Sigmoid slope [1/dB].  When None, uses
                             SIGMOID_SLOPE from config (initial estimate).
                             Pass the value fitted to the Sionna LDPC BER curve.

    Returns
    -------
    float  Packet error rate in [0, 1).
    """
    if tx_eirp_dbm is None:
        tx_eirp_dbm = PHONE_EIRP_DBM
    if snr_thresh_db is None:
        snr_thresh_db = SNR_THRESH_DB
    if sigmoid_slope is None:
        sigmoid_slope = SIGMOID_SLOPE

    # Urban multipath correction: relative RT gain compared to the best
    # satellite.  The reference satellite gets 0 dB correction; others
    # receive a negative correction equal to their RT gain deficit.
    # If RT produced no valid paths (gain = −200 dB), apply a −10 dB penalty.
    gain_for_budget_db = rt_mean_gain_db
    if rt_gain_p10_db is not None:
        gain_for_budget_db = (
            (1.0 - RT_GAIN_P10_BLEND) * rt_mean_gain_db
            + RT_GAIN_P10_BLEND * rt_gain_p10_db
        )

    if gain_for_budget_db <= -150.0:
        urban_correction_db = -10.0    # No paths → deep shadow, penalty
    elif rt_ref_gain_db is not None and rt_ref_gain_db > -150.0:
        urban_correction_db = gain_for_budget_db - rt_ref_gain_db
    else:
        urban_correction_db = 0.0

    # Full uplink budget:
    #   SNR = TX_EIRP − FSPL + urban_correction + SAT_RX_GAIN − NOISE_FLOOR
    # SAT_RX_ANTENNA_GAIN_DB models the satellite's phased-array receive
    # aperture (~30 dBi at 3.5 GHz for a modern LEO NTN spot beam).
    snr_db = (tx_eirp_dbm - fspl_db + urban_correction_db
              + SAT_RX_ANTENNA_GAIN_DB - NOISE_FLOOR_DBM)
    per    = 1.0 / (1.0 + math.exp(sigmoid_slope * (snr_db - snr_thresh_db)))
    return float(np.clip(per, 0.0, 0.99))


# =============================================================================
# Handover schedule computation
# =============================================================================

def _compute_handover_schedule(channel_stats: list,
                                tx_eirp_dbm: float = None,
                                snr_thresh_db: float = None,
                                sigmoid_slope: float = None,
                                rng_seed: int = 42) -> list:
    """
    Determine the sequence of (satellite_id, start_time, end_time,
    per, delay_ms, interruption_ms) intervals for the simulation duration.

    The algorithm:
    1. Sort satellites by decreasing elevation (highest = first to serve).
    2. Assign time slices weighted by slant_range / (v_orb × sin(elev)):
       satellites near the horizon are visible for longer in a real orbital
       pass, so they receive proportionally more simulation time.
    3. Satellites below SAT_HANDOVER_ELEVATION_DEG are skipped (link drop).
    4. Each handover transition is assigned a beam interruption gap drawn
       uniformly from [HANDOVER_INTERRUPTION_MS_MIN, HANDOVER_INTERRUPTION_MS_MAX]
       per 3GPP TS 38.300 §10.1.2.3.

    Parameters
    ----------
    channel_stats : list[dict]
        RT channel statistics, one dict per satellite (output of
        rt_sim.run_ray_tracing()).
    tx_eirp_dbm : float, optional
        Transmitter EIRP passed through to _rt_calibrated_per().
        Defaults to PHONE_EIRP_DBM when None.
    snr_thresh_db : float, optional
        Sigmoid threshold [dB].  Defaults to SNR_THRESH_DB when None.
        Pass the value fitted to the Sionna LDPC BER curve.
    sigmoid_slope : float, optional
        Sigmoid slope [1/dB].  Defaults to SIGMOID_SLOPE when None.
        Pass the value fitted to the Sionna LDPC BER curve.
    rng_seed : int
        Seed for the interruption-gap RNG (reproducibility).

    Returns
    -------
    list of dicts with keys:
        sat_id           int     Satellite index.
        t_start          float   Start of service window [s].
        t_end            float   End of service window [s].
        elev_deg         float   Elevation angle [degrees].
        per              float   Service-link PER (RT-calibrated).
        delay_ms         float   Service-link one-way propagation delay [ms].
        interruption_ms  float   Beam gap before this slot starts [ms].
                                 Zero for the first slot (no incoming HO).
    """
    rng = _random_mod.Random(rng_seed)

    # Sort by elevation, descending (highest elevation served first)
    sorted_stats = sorted(channel_stats, key=lambda s: s["elevation_deg"],
                          reverse=True)

    # Keep only satellites above the handover threshold
    visible = [s for s in sorted_stats
               if s["elevation_deg"] >= SAT_HANDOVER_ELEVATION_DEG]

    if not visible:
        # Fallback: use all satellites even if below threshold
        visible = sorted_stats

    # Slot duration weighting: satellites near the horizon spend more time
    # above the handover threshold during a real orbital pass.
    #
    # The angular rate of elevation change for a satellite on a zenith-pass
    # trajectory is approximately:
    #   d(elevation)/dt ≈ v_orb × sin(elevation) / slant_range(elevation)
    # A satellite at LOW elevation has a SMALL d(elev)/dt → it stays near that
    # elevation for LONGER.  The dwell time per unit elevation is therefore
    # proportional to:
    #   weight ∝ 1 / |d(elev)/dt| ∝ slant_range / (v_orb × sin(elevation))
    #
    # Reference: orbital mechanics (Wertz, "Space Mission Engineering", §9.1;
    #            3GPP TR 38.821 §6.1 orbit geometry).
    elev_rads   = [math.radians(max(s["elevation_deg"], 1.0)) for s in visible]
    raw_weights = [
        _slant_range_m(SAT_HEIGHT_M, max(s["elevation_deg"], 1.0))
        / (SAT_ORBITAL_VELOCITY_MS * math.sin(e))
        for s, e in zip(visible, elev_rads)
    ]
    total_w        = sum(raw_weights)
    slot_durations = [w / total_w * SIM_DURATION_S for w in raw_weights]

    schedule = []

    # Reference gain: the best (highest-elevation) satellite's RT gain.
    # All other satellites are penalised relative to this value.
    ref_gain_db = visible[0]["mean_path_gain_db"] if visible else None

    t_cursor = 0.0
    for i, stat in enumerate(visible):
        slot_s   = slot_durations[i]
        t_start  = t_cursor
        t_end    = t_cursor + slot_s
        t_cursor = t_end
        elev_deg = stat["elevation_deg"]

        fspl   = _fspl_db(SAT_HEIGHT_M, max(elev_deg, 1.0))
        per    = _rt_calibrated_per(fspl, stat["mean_path_gain_db"],
                                    stat.get("mean_path_gain_p10_db"),
                                    ref_gain_db, tx_eirp_dbm,
                                    snr_thresh_db, sigmoid_slope)
        delay  = _one_way_delay_ms(SAT_HEIGHT_M, max(elev_deg, 1.0))

        # Beam interruption gap: zero for the first slot (no incoming HO),
        # random draw for subsequent slots per 3GPP TS 38.300 §10.1.2.3.
        if i == 0:
            gap_ms = 0.0
        else:
            gap_ms = rng.uniform(HANDOVER_INTERRUPTION_MS_MIN,
                                 HANDOVER_INTERRUPTION_MS_MAX)

        schedule.append(dict(
            sat_id         = stat["sat_id"],
            t_start        = round(t_start, 3),
            t_end          = round(t_end, 3),
            elev_deg       = elev_deg,
            per            = round(per, 4),
            delay_ms       = round(delay, 2),
            interruption_ms = round(gap_ms, 1),
        ))

    return schedule


# =============================================================================
# NS-3 topology helpers
# =============================================================================

def _nc2(a, b):
    """NodeContainer from two nodes (cppyy lacks the 2-arg constructor)."""
    nc = ns.NodeContainer()
    nc.Add(a)
    nc.Add(b)
    return nc


def _configure_tcp(protocol_cfg: dict) -> None:
    """
    Apply global NS-3 TCP stack configuration.

    Called once before creating any sockets so that the settings are
    visible to all TCP connections in the simulation.

    Parameters
    ----------
    protocol_cfg : dict  Protocol configuration entry from config.PROTOCOLS
                         augmented with TCP_* keys from config.py.
    """
    variant  = protocol_cfg.get("tcp_variant", "Cubic") or "Cubic"
    snd_buf  = protocol_cfg.get("tcp_snd_buf",  TCP_SNDRCV_BUF_BYTES)
    rcv_buf  = protocol_cfg.get("tcp_rcv_buf",  TCP_SNDRCV_BUF_BYTES)
    sack     = protocol_cfg.get("tcp_sack",     TCP_SACK_ENABLED)
    ts       = protocol_cfg.get("tcp_timestamps", TCP_TIMESTAMPS)

    # Congestion control algorithm
    ns.Config.SetDefault(
        "ns3::TcpL4Protocol::SocketType",
        ns.StringValue(f"ns3::Tcp{variant}"))

    # Socket buffer sizes (critical for NTN high-BDP paths)
    ns.Config.SetDefault(
        "ns3::TcpSocket::SndBufSize",
        ns.UintegerValue(snd_buf))
    ns.Config.SetDefault(
        "ns3::TcpSocket::RcvBufSize",
        ns.UintegerValue(rcv_buf))

    # SACK
    ns.Config.SetDefault(
        "ns3::TcpSocketBase::Sack",
        ns.BooleanValue(bool(sack)))

    # Timestamps
    ns.Config.SetDefault(
        "ns3::TcpSocketBase::Timestamp",
        ns.BooleanValue(bool(ts)))


# =============================================================================
# QUIC post-FlowMonitor corrections
# =============================================================================

def _apply_quic_corrections(bbr_result: dict, schedule: list,
                             link_rate_bps: float = None) -> dict:
    """
    Apply RFC 9000/9002 mechanism corrections on top of a TCP BBR
    FlowMonitor result to produce a QUIC emulation result.

    QUIC is emulated over UDP in NS-3.  This function takes the
    aggregate BBR result (the best TCP baseline for satellite) and
    computes additive throughput/latency corrections for each QUIC
    mechanism that has a measurable effect on a single bulk flow over
    an NTN link.

    Corrections applied
    -------------------
    1. 1-RTT handshake saving
       QUIC completes the handshake in 1 RTT (vs TCP+TLS 1.3 = 2 RTTs),
       saving one RTT worth of bandwidth that TCP wastes in the slow-start
       phase.
       Δtput += (mean_RTT_s × link_rate_bps) / effective_active_s   [kbps]

    2. PTO vs RTO (no cwnd→1 collapse)
       QUIC's Probe Timeout fires 1–2 probe packets without resetting cwnd
       to 1 MSS.  At slots where PER > fast-retransmit threshold (~0.01),
       TCP RTO events repeatedly collapse cwnd to 1; QUIC does not.
       Δtput += cwnd_floor_pkts × MSS × rto_rate × 8 / 1e3   [kbps]
       where rto_rate = estimated RTO events per second based on PER and RTT.
       cwnd_floor_pkts = 2 per RFC 9002 §6.2.4 (minimum 2 packets in flight
       during PTO probing).

    3. Unlimited ACK ranges
       At PER > 0.5, TCP SACK (3 block limit) cannot describe the full loss
       map in one ACK, requiring multiple ACK round trips.  QUIC ACK frames
       support unlimited gap/range pairs → 1 ACK suffices.
       Applied as a latency reduction at high-PER slots.
       Source: IETF draft-kuhn-quic-4-sat Table 1 (Kuhn et al., 2020).

    4. Post-handover throughput dip reduction
       QUIC connection migration (PATH_CHALLENGE/PATH_RESPONSE) costs 1 RTT
       after the beam gap ends and preserves ssthresh.  TCP requires a full
       slow-start (~9 RTTs) after the gap.  Both protocols are subject to
       the same RF blackout gap; only recovery RTTs differ.
       Source: RFC 9000 §9.3.

    5. Loss reduction via PTO vs RTO
       QUIC PTO probes resolve isolated losses without RTO; effective loss
       fraction decays exponentially with RTT relative to the reference RTT.
        Reduction factor = exp(-0.8 × mean_RTT_s / REF_ONE_WAY_DELAY_S) × PER_ratio
        where REF_ONE_WAY_DELAY_S = 0.035 s (35 ms one-way reference delay,
        half of the 70 ms median LEO RTT reported in Sander et al. IMC 2022 and
        IETF draft-kuhn-quic-4-sat Table 1; used as the normalisation denominator).
       PER_ratio = mean(slot PER) over the schedule.

    6. HoL blocking elimination
       NOT applied — zero benefit for a single bulk stream.  HoL blocking
       elimination only matters for multi-stream / HTTP/3 workloads.

    Beam interruption gap
    ---------------------
    The total beam blackout time across all handovers is subtracted from
    the effective active simulation duration:
        total_gap_s = sum(slot["interruption_ms"] for all slots) / 1000
        effective_active_s = active_s - total_gap_s
    Both QUIC and TCP experience the same RF blackout; QUIC recovers in
    1 RTT after the gap ends vs TCP's ~9 RTTs.

    Parameters
    ----------
    bbr_result    : dict   FlowMonitor result dict from a TCP BBR run.
    schedule      : list   Handover schedule (list of slot dicts with per, delay_ms,
                           interruption_ms).
    link_rate_bps : float  Actual service-link rate [bps].  When None, uses
                           10 Mbps (the service link data rate in run_ns3).

    Returns
    -------
    dict  QUIC result dict with corrected throughput_kbps, mean_delay_ms,
          loss_pct, and updated label/protocol fields.
    """
    import copy
    result = copy.deepcopy(bbr_result)
    result["protocol"] = "quic"
    result["label"]    = "QUIC"

    # If BBR got zero throughput the link is effectively dead (e.g. direct
    # topology with phone EIRP is too weak).  QUIC cannot overcome a
    # broken link — return zero throughput with a nominal 1 ms latency.
    if bbr_result["throughput_kbps"] == 0.0 and bbr_result["mean_delay_ms"] == 0.0:
        result["throughput_kbps"] = 0.0
        result["mean_delay_ms"]   = 1.0
        result["loss_pct"]        = 0.0
        print(f"\n  QUIC corrections (from BBR baseline):")
        print(f"    BBR baseline = 0 kbps (link failed) — QUIC result = 0 kbps")
        return result

    if link_rate_bps is None:
        link_rate_bps = 10e6     # 10 Mbps service link

    # RFC 9000 §9.3 post-handover recovery RTT counts
    TCP_RECOVERY_RTTS  = 9    # TCP slow-start + SYN from scratch
    QUIC_RECOVERY_RTTS = 1    # QUIC PATH_CHALLENGE/RESPONSE

    # Reference one-way delay for loss-reduction formula (Sander et al.
    # IMC 2022 — median LEO RTT = 70 ms → one-way = 35 ms; also consistent
    # with IETF draft-kuhn-quic-4-sat Table 1).
    REF_ONE_WAY_DELAY_S = 0.035   # 35 ms one-way

    mss_bytes  = PACKET_SIZE_BYTES
    active_s   = SIM_DURATION_S - 1.0
    mean_rtt_s = max(bbr_result["mean_delay_ms"] / 1e3 * 2.0, 0.001)
    handovers  = bbr_result.get("handovers", 0)

    # Total beam blackout across all handovers (shared by QUIC and TCP)
    total_gap_s = sum(s.get("interruption_ms", 0.0) for s in schedule) / 1000.0
    effective_active_s = max(active_s - total_gap_s, 1.0)

    delta_kbps = 0.0

    # ── 1. 1-RTT handshake saving ─────────────────────────────────────────────
    handshake_saving_kbps = (mean_rtt_s * link_rate_bps) / effective_active_s / 1e3
    delta_kbps += handshake_saving_kbps

    # ── 2. PTO vs RTO: no cwnd collapse ───────────────────────────────────────
    cwnd_floor_pkts = 2   # RFC 9002 §6.2.4: minimum 2 packets during PTO probing
    rto_credit_kbps = 0.0
    for slot in schedule:
        slot_duration_s = slot["t_end"] - slot["t_start"]
        slot_per        = slot["per"]
        if slot_per > 0.03:
            window_pkts = max(int(link_rate_bps * mean_rtt_s / (mss_bytes * 8)), 4)
            rto_rate    = slot_per / (mean_rtt_s * window_pkts + 1e-9)
            slot_credit = cwnd_floor_pkts * mss_bytes * rto_rate * 8 / 1e3
            rto_credit_kbps += slot_credit * (slot_duration_s / effective_active_s)
    delta_kbps += rto_credit_kbps

    # ── 3. Unlimited ACK ranges at high-PER slots ─────────────────────────────
    high_per_fraction = sum(
        (s["t_end"] - s["t_start"]) / effective_active_s
        for s in schedule if s["per"] > 0.5
    )
    latency_reduction_ms = bbr_result["mean_delay_ms"] * (2.0 / 3.0) * high_per_fraction

    # ── 4. Post-handover recovery ─────────────────────────────────────────────
    # Both QUIC and TCP stall for total_gap_s during beam blackout.
    # After the gap: QUIC needs 1 RTT, TCP needs ~9 RTTs to recover.
    if handovers > 0:
        handover_credit_kbps = (
            (TCP_RECOVERY_RTTS - QUIC_RECOVERY_RTTS)
            * mean_rtt_s * link_rate_bps / effective_active_s / 1e3
            * handovers
        )
        delta_kbps += handover_credit_kbps
    else:
        handover_credit_kbps = 0.0

    # ── 5. Loss reduction: PTO prevents cwnd collapse ─────────────────────────
    mean_per = (sum(s["per"] * (s["t_end"] - s["t_start"])
                    for s in schedule) / effective_active_s
                if schedule else 0.0)
    loss_reduction_factor = math.exp(-0.8 * mean_rtt_s / REF_ONE_WAY_DELAY_S) * max(mean_per, 1e-6)
    loss_reduction_factor = min(loss_reduction_factor, 0.99)   # cap at 99%

    # ── Apply corrections ─────────────────────────────────────────────────────
    result["throughput_kbps"] = round(
        bbr_result["throughput_kbps"] + delta_kbps, 2)
    result["mean_delay_ms"]   = round(
        max(1.0, bbr_result["mean_delay_ms"] - latency_reduction_ms), 2)
    result["loss_pct"] = round(
        bbr_result["loss_pct"] * (1.0 - loss_reduction_factor), 3)

    print(f"\n  QUIC corrections (from BBR baseline):")
    print(f"    link_rate        : {link_rate_bps/1e6:.2f} Mbps")
    print(f"    beam gap (total) : {total_gap_s*1e3:.0f} ms  "
          f"(effective active = {effective_active_s:.1f} s)")
    print(f"    handshake saving : +{handshake_saving_kbps:.1f} kbps")
    print(f"    PTO vs RTO       : +{rto_credit_kbps:.1f} kbps")
    print(f"    post-H/O credit  : +{handover_credit_kbps:.1f} kbps  "
          f"(RFC 9000 §9.3: {TCP_RECOVERY_RTTS}-{QUIC_RECOVERY_RTTS} RTTs × {handovers} H/O)")
    print(f"    ACK range lat.   : -{latency_reduction_ms:.2f} ms  "
          f"(draft-kuhn-quic-4-sat Table 1)")
    print(f"    loss reduction   : ×{1.0-loss_reduction_factor:.3f}  "
          f"(exp(-0.8×RTT/{REF_ONE_WAY_DELAY_S*1e3:.0f}ms one-way) × PER={mean_per:.3f})")
    print(f"    total Δtput      : +{delta_kbps:.1f} kbps  →  {result['throughput_kbps']:.0f} kbps")
    print(f"    latency          : {bbr_result['mean_delay_ms']:.1f} ms → {result['mean_delay_ms']:.1f} ms")
    print(f"    loss             : {bbr_result['loss_pct']:.2f}% → {result['loss_pct']:.3f}%")

    return result


# =============================================================================
# Main NS-3 simulation run — multi-client ISL relay topology
# =============================================================================

def run_ns3(scenario: str, protocol_cfg: dict, channel_stats: list,
            snr_thresh_db: float = None,
            sigmoid_slope: float = None) -> dict:
    """
    Run one full NS-3 5G-NTN simulation with:
      - NUM_STATIONARY_CLIENTS + NUM_MOVING_CLIENTS client phones
      - Access satellites (Sat 1 … NUM_SATELLITES-1), each connected via an
        ISL to the benchmark satellite (Sat 0, always highest elevation)
      - If USE_BASE_STATIONS: Phone → gNB → AccessSat → ISL → BenchmarkSat → Server
        Otherwise:            Phone → AccessSat → ISL → BenchmarkSat → Server
      - Moving clients use RandomWaypointMobilityModel
      - TCP BulkSend capped at DATA_VOLUME_MB
      - Mixed traffic profiles: streaming/gaming/texting/voice → UDP CBR,
        bulk → TCP
      - Full beam management: 3-phase link interruption gap per handover
      - Per-second throughput time-series via PacketSink probe
      - Jain's fairness index across per-flow throughputs

    The benchmark satellite (Sat 0) is the measurement node: all traffic
    always flows through it regardless of which access satellite is currently
    serving a client.  This keeps throughput measurements consistent across
    handovers.

    Parameters
    ----------
    scenario : str
        Channel scenario label (e.g. ``"urban"``).  Used for display only.
    protocol_cfg : dict
        Protocol configuration dict from config.PROTOCOLS.
        For QUIC: runs as UDP internally; RFC 9000 corrections applied after.
    channel_stats : list[dict]
        RT channel statistics returned by rt_sim.run_ray_tracing().
    snr_thresh_db : float, optional
        Sigmoid threshold [dB] fitted to the Sionna LDPC BER curve.
    sigmoid_slope : float, optional
        Sigmoid slope [1/dB] fitted to the Sionna LDPC BER curve.

    Returns
    -------
    dict with keys:
        scenario, protocol, label, topology, elevation_deg, svc_delay_ms,
        svc_loss_pct, tx_packets, rx_packets, loss_pct,
        mean_delay_ms, jitter_ms, throughput_kbps, handovers, schedule,
        fairness_index, profile_stats, timeseries.
    """
    import collections
    import math as _math

    proto     = protocol_cfg["protocol"]
    label     = protocol_cfg.get("label", proto.upper())
    ns3_proto = "udp" if proto == "quic" else proto

    # TX EIRP for the ground→satellite service link
    tx_eirp   = GNB_EIRP_DBM if USE_BASE_STATIONS else PHONE_EIRP_DBM
    topo_str  = "indirect (gNB)" if USE_BASE_STATIONS else "direct"

    num_clients = NUM_STATIONARY_CLIENTS + NUM_MOVING_CLIENTS

    print(f"\n[NS-3]  {scenario.upper()}  topology={topo_str}  "
          f"protocol={label}  clients={num_clients}")

    # ── TCP global config ─────────────────────────────────────────────────────
    if ns3_proto == "tcp":
        _configure_tcp(protocol_cfg)

    # ── Handover schedule ─────────────────────────────────────────────────────
    schedule = _compute_handover_schedule(
        channel_stats, tx_eirp_dbm=tx_eirp,
        snr_thresh_db=snr_thresh_db,
        sigmoid_slope=sigmoid_slope,
    )
    print(f"  Handover schedule (EIRP={tx_eirp:.0f} dBm): {len(schedule)} slot(s)")
    for slot in schedule:
        gap_str = f"  gap={slot['interruption_ms']:.0f}ms" if slot["interruption_ms"] > 0 else ""
        print(f"    sat{slot['sat_id']}  {slot['t_start']:.1f}s–{slot['t_end']:.1f}s  "
              f"elev={slot['elev_deg']:.1f}°  delay={slot['delay_ms']:.1f} ms  "
              f"PER={slot['per']:.3f}{gap_str}")

    # Weighted-mean service-link delay (P2P channel delay is immutable)
    if schedule:
        total_dur = schedule[-1]["t_end"] - schedule[0]["t_start"]
        weighted_delay_ms = sum(
            s["delay_ms"] * (s["t_end"] - s["t_start"]) for s in schedule
        ) / max(total_dur, 1e-9)
    else:
        weighted_delay_ms = _one_way_delay_ms(SAT_HEIGHT_M, 60.0)

    first = schedule[0] if schedule else dict(
        delay_ms=weighted_delay_ms, per=0.01, elev_deg=60.0,
        sat_id=0, interruption_ms=0.0,
    )
    print(f"  Service link delay (weighted-mean): {weighted_delay_ms:.2f} ms")
    print(f"  ISL (access→benchmark sat):  rate={ISL_DATARATE}  "
          f"delay={ISL_DELAY_MS} ms  PER={ISL_PER}")
    print(f"  Sat→Server:  rate={SAT_SERVER_DATARATE}  "
          f"delay={TERRESTRIAL_BACKHAUL_DELAY_MS:.0f} ms")

    # =========================================================================
    # Node creation
    # =========================================================================
    # Node layout (all in one NodeContainer for InternetStack):
    #   [0 … num_clients-1]               : phone nodes
    #   [num_clients … num_clients+NUM_GNB-1] : gNB nodes (if USE_BASE_STATIONS)
    #   [base_idx]                         : benchmark satellite (Sat 0)
    #   [base_idx+1 … base_idx+n_access]   : access satellites (Sat 1…N)
    #   [base_idx+n_access+1]              : internet server
    #
    # n_access = number of access satellites = len(schedule) - 1 (Sat 0 is
    # the benchmark; we need at least 1 access sat even if only 1 in schedule)

    n_access_sats = max(1, len(schedule) - 1)  # access sats (not the benchmark)
    n_gnbs        = NUM_GNB if USE_BASE_STATIONS else 0

    # Index helpers
    phone_idx   = lambda i: i                                  # 0..num_clients-1
    gnb_idx     = lambda i: num_clients + i                    # 0..n_gnbs-1
    bench_idx   = num_clients + n_gnbs                         # benchmark sat
    access_idx  = lambda i: bench_idx + 1 + i                  # 0..n_access_sats-1
    server_idx  = bench_idx + 1 + n_access_sats
    total_nodes = server_idx + 1

    nodes = ns.NodeContainer()
    nodes.Create(total_nodes)
    ns.InternetStackHelper().Install(nodes)

    phones   = [nodes.Get(phone_idx(i)) for i in range(num_clients)]
    gnbs     = [nodes.Get(gnb_idx(i))   for i in range(n_gnbs)]
    bench    = nodes.Get(bench_idx)
    accesses = [nodes.Get(access_idx(i)) for i in range(n_access_sats)]
    server   = nodes.Get(server_idx)

    p2p = ns.PointToPointHelper()

    # ── Client positions ──────────────────────────────────────────────────────
    # Phones are placed in a circle of CLIENT_AREA_RADIUS_M.
    # Stationary: ConstantPosition.  Moving: RandomWaypoint.
    _rng = _random_mod.Random(42)

    def _random_pos_in_circle(radius, z=1.5):
        """Uniform random (x,y) within a circle of given radius, height z."""
        angle  = _rng.uniform(0, 2 * _math.pi)
        r      = radius * _math.sqrt(_rng.uniform(0, 1))
        return r * _math.cos(angle), r * _math.sin(angle), z

    static_mob = ns.MobilityHelper()
    static_mob.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    static_mob.Install(nodes)   # default all nodes to static first

    # Place stationary phones at fixed random positions
    phone_positions = []
    for i in range(num_clients):
        px, py, pz = _random_pos_in_circle(CLIENT_AREA_RADIUS_M)
        phone_positions.append((px, py, pz))
        mob = phones[i].GetObject[ns.ConstantPositionMobilityModel]()
        mob.SetPosition(ns.Vector(px, py, pz))

    # Override moving phones with RandomWaypointMobilityModel
    # NS-3 RandomWaypointMobilityModel has three attributes: Speed, Pause,
    # PositionAllocator.  There is no "Bounds" attribute — bounds are enforced
    # entirely via RandomRectanglePositionAllocator (X and Y random variables).
    if NUM_PEDESTRIAN_MOVING_CLIENTS + NUM_VEHICULAR_MOVING_CLIENTS != NUM_MOVING_CLIENTS:
        raise ValueError(
            "NUM_PEDESTRIAN_MOVING_CLIENTS + NUM_VEHICULAR_MOVING_CLIENTS "
            "must equal NUM_MOVING_CLIENTS"
        )

    moving_start = NUM_STATIONARY_CLIENTS
    moving_end = NUM_STATIONARY_CLIENTS + NUM_MOVING_CLIENTS
    pedestrian_end = moving_start + NUM_PEDESTRIAN_MOVING_CLIENTS

    def _speed_bounds_for_client(client_idx: int) -> tuple:
        if client_idx < pedestrian_end:
            return PEDESTRIAN_SPEED_MIN_MS, PEDESTRIAN_SPEED_MAX_MS
        return VEHICULAR_SPEED_MIN_MS, VEHICULAR_SPEED_MAX_MS

    for i in range(NUM_STATIONARY_CLIENTS, num_clients):
        px, py, pz = phone_positions[i]
        rwp_mob = ns.CreateObject[ns.RandomWaypointMobilityModel]()
        r = CLIENT_AREA_RADIUS_M
        vmin, vmax = _speed_bounds_for_client(i)
        speed_var = ns.StringValue(
            f"ns3::UniformRandomVariable[Min={vmin}|Max={vmax}]"
        )
        pause_var = ns.StringValue("ns3::ConstantRandomVariable[Constant=0]")
        pos_alloc = ns.CreateObject[ns.RandomRectanglePositionAllocator]()
        pos_alloc.SetAttribute("X", ns.StringValue(
            f"ns3::UniformRandomVariable[Min={-r}|Max={r}]"))
        pos_alloc.SetAttribute("Y", ns.StringValue(
            f"ns3::UniformRandomVariable[Min={-r}|Max={r}]"))
        rwp_mob.SetAttribute("Speed",             speed_var)
        rwp_mob.SetAttribute("Pause",             pause_var)
        rwp_mob.SetAttribute("PositionAllocator", ns.PointerValue(pos_alloc))
        phones[i].AggregateObject(rwp_mob)
        rwp_mob.SetPosition(ns.Vector(px, py, pz))

    # ── Satellite mobility: benchmark at zenith, access sats in orbit ─────────
    sat_mob_b = ns.CreateObject[ns.ConstantVelocityMobilityModel]()
    bench.AggregateObject(sat_mob_b)
    sat_mob_b.SetPosition(ns.Vector(0.0, 0.0, SAT_HEIGHT_M))
    sat_mob_b.SetVelocity(ns.Vector(SAT_ORBITAL_VELOCITY_MS, 0.0, 0.0))

    for i, acc in enumerate(accesses):
        sat_mob_a = ns.CreateObject[ns.ConstantVelocityMobilityModel]()
        acc.AggregateObject(sat_mob_a)
        offset = (i + 1) * 144_000.0   # ~144 km spacing at 550 km
        sat_mob_a.SetPosition(ns.Vector(offset, 0.0, SAT_HEIGHT_M))
        sat_mob_a.SetVelocity(ns.Vector(SAT_ORBITAL_VELOCITY_MS, 0.0, 0.0))

    # ── IP address allocator ──────────────────────────────────────────────────
    ipv4 = ns.Ipv4AddressHelper()
    subnet_iter = [0]  # mutable counter for subnet numbering

    def _next_subnet():
        n = subnet_iter[0]
        subnet_iter[0] += 1
        return f"10.{n // 256}.{n % 256}.0"

    # ── Assign each phone to the nearest gNB (or directly to access sat 0) ───
    # client_to_gnb[i] = gNB index for client i  (USE_BASE_STATIONS only)
    client_to_gnb = []
    if USE_BASE_STATIONS and n_gnbs > 0:
        for i in range(num_clients):
            px, py, _ = phone_positions[i]
            dists = [
                _math.hypot(px - GNB_POSITIONS[g][0], py - GNB_POSITIONS[g][1])
                for g in range(n_gnbs)
            ]
            client_to_gnb.append(int(dists.index(min(dists))))
    else:
        client_to_gnb = [0] * num_clients

    # ── gNB positions ─────────────────────────────────────────────────────────
    for gi, (gx, gy) in enumerate(GNB_POSITIONS[:n_gnbs]):
        gmob = gnbs[gi].GetObject[ns.ConstantPositionMobilityModel]()
        gmob.SetPosition(ns.Vector(gx, gy, 30.0))

    # =========================================================================
    # Wiring up links
    # =========================================================================

    # All phone↔gNB or phone↔access-sat error models stored per-client
    # so handover callback can update each one.
    em_svc_list   = []   # service-link error model per client/gNB
    devs_svc_list = []   # service-link DeviceContainer per client/gNB

    # ── Phone ↔ gNB  (or direct Phone ↔ access-sat-0) ────────────────────────
    if USE_BASE_STATIONS:
        # Phone → gNB  (terrestrial, high data rate, low PER)
        gnb_used = set(client_to_gnb)
        p2p.SetDeviceAttribute("DataRate", ns.StringValue(GNB_DATARATE))
        p2p.SetChannelAttribute("Delay",
            ns.StringValue(f"{GNB_PROCESSING_DELAY_MS:.3f}ms"))
        for i in range(num_clients):
            devs_ug = p2p.Install(_nc2(phones[i], gnbs[client_to_gnb[i]]))
            em_ug = ns.CreateObject[ns.RateErrorModel]()
            em_ug.SetAttribute("ErrorRate", ns.DoubleValue(GNB_TERRESTRIAL_PER))
            em_ug.SetAttribute("ErrorUnit", ns.StringValue("ERROR_UNIT_PACKET"))
            devs_ug.Get(1).SetAttribute("ReceiveErrorModel",
                                        ns.PointerValue(em_ug))
            ipv4.SetBase(ns.Ipv4Address(_next_subnet()),
                         ns.Ipv4Mask("255.255.255.0"))
            ipv4.Assign(devs_ug)

        # gNB → access satellite 0  (NTN hop, GNB EIRP)
        # All gNBs share the same access sat (Sat 0 in schedule order →
        # accesses[0]).  Each gNB gets its own P2P link; they all share
        # the same error model so handover updates all simultaneously.
        p2p.SetDeviceAttribute("DataRate", ns.StringValue("10Mbps"))
        p2p.SetChannelAttribute("Delay",
            ns.StringValue(f"{weighted_delay_ms:.3f}ms"))
        for gi in sorted(gnb_used):
            devs_svc = p2p.Install(_nc2(gnbs[gi], accesses[0]))
            em = ns.CreateObject[ns.RateErrorModel]()
            em.SetAttribute("ErrorRate", ns.DoubleValue(first["per"]))
            em.SetAttribute("ErrorUnit", ns.StringValue("ERROR_UNIT_PACKET"))
            devs_svc.Get(1).SetAttribute("ReceiveErrorModel",
                                          ns.PointerValue(em))
            em_svc_list.append(em)
            devs_svc_list.append(devs_svc)
            ipv4.SetBase(ns.Ipv4Address(_next_subnet()),
                         ns.Ipv4Mask("255.255.255.0"))
            ipv4.Assign(devs_svc)

    else:
        # Direct: Phone → access satellite 0
        p2p.SetDeviceAttribute("DataRate", ns.StringValue("10Mbps"))
        p2p.SetChannelAttribute("Delay",
            ns.StringValue(f"{weighted_delay_ms:.3f}ms"))
        for i in range(num_clients):
            devs_svc = p2p.Install(_nc2(phones[i], accesses[0]))
            em = ns.CreateObject[ns.RateErrorModel]()
            em.SetAttribute("ErrorRate", ns.DoubleValue(first["per"]))
            em.SetAttribute("ErrorUnit", ns.StringValue("ERROR_UNIT_PACKET"))
            devs_svc.Get(1).SetAttribute("ReceiveErrorModel",
                                          ns.PointerValue(em))
            em_svc_list.append(em)
            devs_svc_list.append(devs_svc)
            ipv4.SetBase(ns.Ipv4Address(_next_subnet()),
                         ns.Ipv4Mask("255.255.255.0"))
            ipv4.Assign(devs_svc)

    # ── Access satellite(s) → Benchmark satellite (ISL) ──────────────────────
    p2p.SetDeviceAttribute("DataRate", ns.StringValue(ISL_DATARATE))
    p2p.SetChannelAttribute("Delay",   ns.StringValue(f"{ISL_DELAY_MS:.3f}ms"))
    em_isl_list = []
    for acc in accesses:
        devs_isl = p2p.Install(_nc2(acc, bench))
        em_isl = ns.CreateObject[ns.RateErrorModel]()
        em_isl.SetAttribute("ErrorRate", ns.DoubleValue(ISL_PER))
        em_isl.SetAttribute("ErrorUnit", ns.StringValue("ERROR_UNIT_PACKET"))
        devs_isl.Get(1).SetAttribute("ReceiveErrorModel",
                                      ns.PointerValue(em_isl))
        em_isl_list.append(em_isl)
        ipv4.SetBase(ns.Ipv4Address(_next_subnet()),
                     ns.Ipv4Mask("255.255.255.0"))
        ipv4.Assign(devs_isl)

    # ── Benchmark satellite → Internet Server (direct link) ───────────────────
    p2p.SetDeviceAttribute("DataRate",  ns.StringValue(SAT_SERVER_DATARATE))
    p2p.SetChannelAttribute("Delay",
        ns.StringValue(f"{TERRESTRIAL_BACKHAUL_DELAY_MS:.0f}ms"))
    devs_srv = p2p.Install(_nc2(bench, server))
    ipv4.SetBase(ns.Ipv4Address(_next_subnet()), ns.Ipv4Mask("255.255.255.0"))
    iface_srv = ipv4.Assign(devs_srv)

    ns.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

    # =========================================================================
    # Beam management handover scheduling
    # =========================================================================
    # Each handover has two scheduled events:
    #   Event A  at slot["t_start"]                    → blackout (PER = 1.0)
    #   Event B  at slot["t_start"] + gap_ms / 1000   → restore  (PER = slot["per"])
    #
    # All service-link error models in em_svc_list are updated together
    # because all clients are within CLIENT_AREA_RADIUS_M of each other,
    # which is negligible vs the satellite footprint (~100 km).

    handover_queue = collections.deque(schedule[1:])
    handover_count = [0]
    handover_times = []   # list of (t_blackout_start, t_blackout_end) tuples

    def _handover_blackout():
        """Phase 1: set PER = 1.0 (beam failure / link blackout)."""
        if not handover_queue:
            return
        slot   = handover_queue[0]   # peek — don't pop yet
        gap_s  = slot["interruption_ms"] / 1000.0
        t_now  = ns.Simulator.Now().GetSeconds()
        handover_times.append((t_now, t_now + gap_s))
        print(f"  [t={t_now:.2f}s] Handover blackout → sat{slot['sat_id']}  "
              f"gap={slot['interruption_ms']:.0f} ms  "
              f"elev={slot['elev_deg']:.1f}°")
        # Set all service-link error models to 1.0 (full drop)
        for em in em_svc_list:
            em.SetAttribute("ErrorRate", ns.DoubleValue(1.0))
        # Schedule the restore event after the gap
        ev = ns.cppyy.gbl.pythonMakeEvent(_handover_restore)
        ns.Simulator.Schedule(ns.Seconds(gap_s), ev)

    def _handover_restore():
        """Phase 2: restore PER = slot['per'] after beam management gap."""
        if not handover_queue:
            return
        slot = handover_queue.popleft()
        for em in em_svc_list:
            em.SetAttribute("ErrorRate", ns.DoubleValue(slot["per"]))
        handover_count[0] += 1
        print(f"  [t={ns.Simulator.Now().GetSeconds():.2f}s] "
              f"Handover restored → sat{slot['sat_id']}  "
              f"PER={slot['per']:.3f}")
        # Schedule the next handover blackout if one exists
        if handover_queue:
            nxt = handover_queue[0]
            if nxt["t_start"] < SIM_DURATION_S:
                ev = ns.cppyy.gbl.pythonMakeEvent(_handover_blackout)
                ns.Simulator.Schedule(
                    ns.Seconds(nxt["t_start"] - ns.Simulator.Now().GetSeconds()),
                    ev)

    if len(schedule) > 1:
        first_ho = schedule[1]
        if first_ho["t_start"] < SIM_DURATION_S:
            ev = ns.cppyy.gbl.pythonMakeEvent(_handover_blackout)
            ns.Simulator.Schedule(ns.Seconds(first_ho["t_start"]), ev)

    # =========================================================================
    # Application layer — mixed traffic profiles
    # =========================================================================
    port        = 9
    server_addr = ns.InetSocketAddress(iface_srv.GetAddress(1), port)

    # Compute per-flow MaxBytes for TCP BulkSend
    max_bytes = 0  # unlimited (for UDP; ignored)
    if DATA_VOLUME_MB > 0:
        max_bytes = int(DATA_VOLUME_MB * 1_000_000)

    # Install per-client applications according to CLIENT_PROFILES.
    # streaming / gaming / texting / voice → OnOffHelper (UDP CBR)
    # bulk                 → BulkSendHelper (TCP)
    #
    # When the protocol under test is QUIC (ns3_proto == "udp") or plain UDP,
    # bulk clients also use OnOff/UDP instead of BulkSend/TCP because the
    # protocol comparison must be consistent (all flows use the same socket
    # factory).  TCP bulk profiles are only active when ns3_proto == "tcp".

    udp_sink = ns.PacketSinkHelper(
        "ns3::UdpSocketFactory",
        ns.InetSocketAddress(ns.Ipv4Address.GetAny(), port).ConvertTo())
    tcp_sink = ns.PacketSinkHelper(
        "ns3::TcpSocketFactory",
        ns.InetSocketAddress(ns.Ipv4Address.GetAny(), port).ConvertTo())

    # Install sinks on server before apps on phones
    udp_sink_app = udp_sink.Install(server)
    udp_sink_app.Start(ns.Seconds(0.0))
    udp_sink_app.Stop(ns.Seconds(SIM_DURATION_S + 2.0))

    tcp_sink_app = None
    if ns3_proto == "tcp":
        tcp_sink_app = tcp_sink.Install(server)
        tcp_sink_app.Start(ns.Seconds(0.0))
        tcp_sink_app.Stop(ns.Seconds(SIM_DURATION_S + 2.0))

    # Track which sink app handles which ip_proto_num for FlowMonitor
    # We collect stats for both protocols (6 + 17) in a single pass.
    active_ip_protos = {17}   # UDP always present
    if ns3_proto == "tcp":
        active_ip_protos.add(6)

    for ci, phone in enumerate(phones):
        profile_name = CLIENT_PROFILES[ci] if ci < len(CLIENT_PROFILES) else "bulk"
        profile      = TRAFFIC_PROFILES[profile_name]
        app_type     = profile["app_type"]

        # When running a non-TCP protocol comparison pass (UDP/QUIC), treat
        # all clients as UDP CBR with the profile's data_rate.  This ensures
        # every protocol comparison run uses the same number of active flows.
        if ns3_proto != "tcp" or app_type != "tcp_bulk":
            # UDP OnOff application
            data_rate  = profile.get("data_rate") or APP_DATA_RATE
            pkt_size   = profile["packet_size"]
            duty       = profile["duty"]

            if duty >= 1.0:
                on_time_str  = "ns3::ConstantRandomVariable[Constant=1]"
                off_time_str = "ns3::ConstantRandomVariable[Constant=0]"
            else:
                # IoT duty cycle: on for ~10% of time
                on_str       = f"ns3::UniformRandomVariable[Min=0.9|Max=1.1]"
                off_period   = (1.0 / max(duty, 0.01)) - 1.0
                off_str      = (f"ns3::UniformRandomVariable"
                                f"[Min={off_period*0.9:.2f}|Max={off_period*1.1:.2f}]")
                on_time_str  = on_str
                off_time_str = off_str

            onoff = ns.OnOffHelper(
                "ns3::UdpSocketFactory", server_addr.ConvertTo())
            onoff.SetAttribute("DataRate",   ns.StringValue(data_rate))
            onoff.SetAttribute("PacketSize", ns.UintegerValue(pkt_size))
            onoff.SetAttribute("OnTime",     ns.StringValue(on_time_str))
            onoff.SetAttribute("OffTime",    ns.StringValue(off_time_str))
            for _ in range(NUM_PARALLEL_FLOWS):
                app_tx = onoff.Install(phone)
                app_tx.Start(ns.Seconds(1.0))
                app_tx.Stop(ns.Seconds(SIM_DURATION_S))

        else:
            # TCP BulkSend application (bulk profile, TCP protocol run only)
            bulk = ns.BulkSendHelper(
                "ns3::TcpSocketFactory", server_addr.ConvertTo())
            bulk.SetAttribute("SendSize", ns.UintegerValue(profile["packet_size"]))
            bulk.SetAttribute("MaxBytes", ns.UintegerValue(max_bytes))
            for _ in range(NUM_PARALLEL_FLOWS):
                app_tx = bulk.Install(phone)
                app_tx.Start(ns.Seconds(1.0))
                app_tx.Stop(ns.Seconds(SIM_DURATION_S))

    # =========================================================================
    # Per-second throughput time-series probe (PacketSink cumulative RX bytes)
    # =========================================================================
    # The probe reads the total received bytes from the server's UDP sink
    # (and TCP sink when present) every TIMESERIES_BUCKET_S seconds and
    # records the delta as per-second throughput.
    #
    # NOTE: For simplicity we probe the UDP sink only when ns3_proto == "udp",
    # and the TCP sink only when ns3_proto == "tcp".  In mixed-profile runs
    # (ns3_proto == "tcp") the UDP flows from streaming/gaming/texting/voice
    # clients will not appear in the TCP PacketSink — this is intentional; we only
    # time-series the primary protocol under test.

    ts_t_list     = []   # simulation time at each probe [s]
    ts_bytes_list = []   # cumulative RX bytes at each probe

    # Get the primary sink app handle (index 0 = first installed)
    primary_sink_app = tcp_sink_app.Get(0) if (ns3_proto == "tcp" and tcp_sink_app is not None) else udp_sink_app.Get(0)

    def _ts_probe():
        t        = ns.Simulator.Now().GetSeconds()
        cum_bytes = primary_sink_app.GetObject[ns.PacketSink]().GetTotalRx()
        ts_t_list.append(t)
        ts_bytes_list.append(int(cum_bytes))
        # Reschedule until end of simulation
        if t + TIMESERIES_BUCKET_S <= SIM_DURATION_S + 1.0:
            ev = ns.cppyy.gbl.pythonMakeEvent(_ts_probe)
            ns.Simulator.Schedule(ns.Seconds(TIMESERIES_BUCKET_S), ev)

    # Start first probe at t = TIMESERIES_BUCKET_S
    ev_ts = ns.cppyy.gbl.pythonMakeEvent(_ts_probe)
    ns.Simulator.Schedule(ns.Seconds(TIMESERIES_BUCKET_S), ev_ts)

    # ── FlowMonitor ───────────────────────────────────────────────────────────
    fm_helper = ns.FlowMonitorHelper()
    endpoint_nc = ns.NodeContainer()
    for ph in phones:
        endpoint_nc.Add(ph)
    endpoint_nc.Add(server)
    monitor = fm_helper.Install(endpoint_nc)

    # ── Run simulation ────────────────────────────────────────────────────────
    ns.Simulator.Stop(ns.Seconds(SIM_DURATION_S + 3.0))
    ns.Simulator.Run()
    monitor.CheckForLostPackets()

    # ── Build per-second time-series ──────────────────────────────────────────
    ts_throughput_kbps = []
    for k in range(len(ts_t_list)):
        prev_bytes = ts_bytes_list[k - 1] if k > 0 else 0
        delta_bytes = max(ts_bytes_list[k] - prev_bytes, 0)
        ts_throughput_kbps.append(round(delta_bytes * 8.0 / TIMESERIES_BUCKET_S / 1e3, 2))

    timeseries = {
        "t_s":              [round(t, 1) for t in ts_t_list],
        "throughput_kbps":  ts_throughput_kbps,
        "handover_times":   list(handover_times),
    }

    # ── Collect FlowMonitor statistics ────────────────────────────────────────
    stats      = monitor.GetFlowStats()
    classifier = fm_helper.GetClassifier()

    result = dict(
        scenario        = scenario,
        protocol        = proto,
        label           = label,
        topology        = topo_str,
        elevation_deg   = first["elev_deg"],
        svc_delay_ms    = round(weighted_delay_ms, 2),
        svc_loss_pct    = round(first["per"] * 100.0, 2),
        tx_packets      = 0,
        rx_packets      = 0,
        loss_pct        = 0.0,
        mean_delay_ms   = 0.0,
        jitter_ms       = 0.0,
        throughput_kbps = 0.0,
        handovers       = handover_count[0],
        schedule        = schedule,
        fairness_index  = 0.0,
        profile_stats   = {p: {"tx": 0, "rx": 0, "rx_bytes": 0, "delay_sum": 0.0}
                           for p in TRAFFIC_PROFILES},
        timeseries      = timeseries,
    )

    active_s = SIM_DURATION_S - 1.0   # exclude 1 s warm-up

    total_rx   = 0
    total_tx   = 0
    delay_sum  = 0.0
    jitter_sum = 0.0
    rx_bytes   = 0
    flow_tputs = []   # per-flow throughput for Jain's fairness

    # Map flow source port range to client index for profile tagging.
    # NS-3 FlowMonitor provides source/dest address but not the client index
    # directly.  We use a simpler approach: tag by IP protocol number.
    # UDP flows → distributed over UDP CBR profile names in proportion.
    # TCP flows → bulk profile.
    udp_flow_count = 0
    tcp_flow_count = 0

    for pair in stats:
        fid, fs = pair.first, pair.second
        finfo   = classifier.FindFlow(fid)
        ip_proto = finfo.protocol

        if ip_proto not in active_ip_protos:
            continue
        if fs.rxPackets == 0:
            continue

        rx_n   = int(fs.rxPackets)
        tx_n   = int(fs.txPackets)
        d_s    = fs.delaySum.GetSeconds()
        j_s    = fs.jitterSum.GetSeconds()
        b      = int(fs.rxBytes)
        active_dur = max(
            fs.timeLastRxPacket.GetSeconds() - fs.timeFirstRxPacket.GetSeconds(),
            1e-9)
        flow_tput = b * 8.0 / active_dur / 1e3  # kbps for this flow

        total_rx   += rx_n
        total_tx   += tx_n
        delay_sum  += d_s
        jitter_sum += j_s
        rx_bytes   += b
        flow_tputs.append(flow_tput)

        # Profile tagging: TCP flows → bulk; UDP flows → cycle through
        # configured UDP profile names in rough proportion.
        if ip_proto == 6:
            tcp_flow_count += 1
            ps = result["profile_stats"]["bulk"]
            ps["tx"] += tx_n
            ps["rx"] += rx_n
            ps["rx_bytes"] += b
            ps["delay_sum"] += d_s
        else:
            # Assign UDP flows to all configured UDP profiles in round-robin
            udp_profiles = [
                pname for pname, pdef in TRAFFIC_PROFILES.items()
                if pdef.get("app_type") == "udp_cbr"
            ]
            if not udp_profiles:
                udp_profiles = ["streaming"]
            pname = udp_profiles[udp_flow_count % len(udp_profiles)]
            udp_flow_count += 1
            ps = result["profile_stats"][pname]
            ps["tx"] += tx_n
            ps["rx"] += rx_n
            ps["rx_bytes"] += b
            ps["delay_sum"] += d_s

    if total_rx > 0:
        # Jain's fairness index: J = (Σx)² / (n · Σx²)
        n_flows = len(flow_tputs)
        if n_flows > 0 and sum(x**2 for x in flow_tputs) > 0:
            fairness = (sum(flow_tputs) ** 2) / (n_flows * sum(x**2 for x in flow_tputs))
        else:
            fairness = 0.0

        result.update(
            tx_packets      = total_tx,
            rx_packets      = total_rx,
            loss_pct        = round(100.0 * (1 - total_rx / max(total_tx, 1)), 2),
            mean_delay_ms   = round(delay_sum / total_rx * 1e3, 2),
            jitter_ms       = round(jitter_sum / max(total_rx - 1, 1) * 1e3, 2),
            throughput_kbps = round(rx_bytes * 8.0 / active_s / 1e3, 2),
            fairness_index  = round(fairness, 4),
        )

    ns.Simulator.Destroy()

    print(f"\n  Results ({label}, {topo_str}):")
    for k, v in result.items():
        if k not in ("schedule", "profile_stats", "timeseries"):
            print(f"    {k:<24s}: {v}")
    print(f"    {'fairness_index':<24s}: {result['fairness_index']:.4f}")

    return result


# =============================================================================
# Protocol suite runner — returns (results, results) for backward compat
# =============================================================================

def run_ns3_both_topologies(channel_stats: list,
                             scenario: str = "urban",
                             snr_thresh_db: float = None,
                             sigmoid_slope: float = None) -> tuple:
    """
    Run the NS-3 simulation once per entry in config.PROTOCOLS using the
    topology mode configured in config.USE_BASE_STATIONS, then return the
    results as a (direct_results, indirect_results) tuple for backward
    compatibility with main.py and topology_diagram.py.

    Because the topology is now determined by USE_BASE_STATIONS (a single
    configurable flag), the same result set is returned in both slots of
    the tuple.  All plot functions in topology_diagram.py continue to
    work without modification.

    For QUIC: the NS-3 simulation runs as UDP.  After the BBR run completes,
    RFC 9000 corrections are applied analytically via _apply_quic_corrections.

    Parameters
    ----------
    channel_stats : list[dict]
        RT channel statistics from rt_sim.run_ray_tracing().
    scenario : str
        NTN propagation scenario label (used for display only).
    snr_thresh_db : float, optional
        Sigmoid threshold [dB] fitted to the Sionna LDPC BER curve.
    sigmoid_slope : float, optional
        Sigmoid slope [1/dB] fitted to the Sionna LDPC BER curve.

    Returns
    -------
    tuple (results, results)
        Both elements contain the same list[dict], one entry per protocol.
        The tuple structure is kept for backward compatibility with callers
        that unpack as:  direct_results, indirect_results = run_ns3_both_topologies(...)
    """
    results             = []
    bbr_result_for_quic = None

    for pcfg in PROTOCOLS:
        if pcfg["protocol"] == "quic":
            continue   # defer until BBR is done

        full_cfg = dict(
            tcp_snd_buf    = TCP_SNDRCV_BUF_BYTES,
            tcp_rcv_buf    = TCP_SNDRCV_BUF_BYTES,
            tcp_sack       = TCP_SACK_ENABLED,
            tcp_timestamps = TCP_TIMESTAMPS,
            **pcfg,
        )
        r = run_ns3(scenario, full_cfg, channel_stats,
                    snr_thresh_db=snr_thresh_db, sigmoid_slope=sigmoid_slope)
        results.append(r)

        if pcfg.get("label") == "TCP BBR":
            bbr_result_for_quic = r

    # QUIC: analytical corrections on top of BBR baseline
    if bbr_result_for_quic is not None:
        quic_result = _apply_quic_corrections(
            bbr_result_for_quic,
            bbr_result_for_quic["schedule"],
            link_rate_bps=10e6,   # 10 Mbps service link
        )
        topo_str = "indirect (gNB)" if USE_BASE_STATIONS else "direct"
        quic_result["topology"] = topo_str
        results.append(quic_result)
    else:
        print("[QUIC]  No BBR result found — skipping QUIC.")

    # Return the same result set in both slots for backward compatibility
    return results, results


# =============================================================================
# Legacy single-topology runner (backward compatibility)
# =============================================================================

def run_ns3_protocol_suite(channel_stats: list,
                            scenario: str = "urban") -> list:
    """Kept for backward compatibility. Use run_ns3_both_topologies instead."""
    results, _ = run_ns3_both_topologies(channel_stats, scenario)
    return results
