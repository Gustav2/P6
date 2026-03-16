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

Topologies
----------

  Direct (4 nodes):
    Phone (UE)
       │  5G-NR NTN service link  (550 km LEO, RT-calibrated PER, PHONE_EIRP)
       ▼
    Satellite [moving]     ← handover when elevation < SAT_HANDOVER_ELEVATION_DEG
       │  Ka-band feeder link  (100 Mbps, ~2 ms)
       ▼
    Ground Station
       │  Terrestrial fibre  (1 Gbps, 10 ms)
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
       │  Ka-band feeder link  (100 Mbps, ~2 ms)
       ▼
    Ground Station
       │  Terrestrial fibre  (1 Gbps, 10 ms)
       ▼
    Internet Server

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
                                 is reduced by 50 % vs TCP BBR.

  5. HoL blocking:              NOT applied — zero benefit for single bulk
                                 stream (only matters for multi-stream HTTP/3).

The QUIC baseline is TCP BBR (BBR is the congestion controller used by
real QUIC implementations such as Chromium/quiche and lsquic for satellite
links).  Corrections are additive on top of the BBR aggregate result.

Satellite mobility
------------------
Each satellite node is given a ConstantVelocity mobility model.
NS-3 does not enforce signal interruption based on position, so the
handover is modelled by scheduling link-rate change callbacks at handover
times using a single deque-popping function (cppyy limitation workaround).

Dependencies
------------
  NS-3 with Python bindings (cppyy-based, 3.37+).
  See README.md for build instructions.
"""

import math
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
    FEEDER_FREQ_HZ,
    FEEDER_BANDWIDTH_HZ,
    GS_RX_ANTENNA_GAIN_DB,
    SAT_TX_EIRP_FEEDER_DBM,
    NOISE_FLOOR_DBM,
    SNR_THRESH_DB,
    SIGMOID_SLOPE,
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
    if rt_mean_gain_db <= -150.0:
        urban_correction_db = -10.0    # No paths → deep shadow, penalty
    elif rt_ref_gain_db is not None and rt_ref_gain_db > -150.0:
        urban_correction_db = rt_mean_gain_db - rt_ref_gain_db
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


def _feeder_snr_db(cs: dict, ref_feeder_gain_db: float) -> float:
    """
    Compute the feeder-link SNR [dB] for a satellite entry.

    Uses Ka-band FSPL (FEEDER_FREQ_HZ) over the real 550 km slant range
    plus the RT-derived urban correction (relative path-gain difference
    between this satellite and the reference/best satellite).

    Parameters
    ----------
    cs                : dict   Channel stats entry with feeder_* keys.
    ref_feeder_gain_db: float  RT feeder gain of the best satellite [dB].
                               Pass the maximum feeder_mean_path_gain_db
                               across all valid satellites.

    Returns
    -------
    float  SNR in dB.
    """
    elev  = max(cs["feeder_elevation_deg"], 1.0)
    fspl  = _fspl_db(SAT_HEIGHT_M, elev, fc_hz=FEEDER_FREQ_HZ)
    gain  = cs["feeder_mean_path_gain_db"]

    if gain <= -150.0:
        urban = -10.0           # No RT paths → deep shadow penalty
    elif ref_feeder_gain_db > -150.0:
        urban = gain - ref_feeder_gain_db
    else:
        urban = 0.0

    return (SAT_TX_EIRP_FEEDER_DBM - fspl + urban
            + GS_RX_ANTENNA_GAIN_DB - NOISE_FLOOR_DBM)


def _feeder_calibrated_per(cs: dict, ref_feeder_gain_db: float,
                            snr_thresh_db: float = None,
                            sigmoid_slope: float = None) -> float:
    """
    Sigmoid PER model for the Satellite→GS feeder link.

    Mirrors _rt_calibrated_per but uses Ka-band FSPL (FEEDER_FREQ_HZ),
    the satellite feeder EIRP (SAT_TX_EIRP_FEEDER_DBM), and the GS dish
    gain (GS_RX_ANTENNA_GAIN_DB) instead of service-link parameters.

    Parameters
    ----------
    cs                : dict   Channel stats entry with feeder_* keys.
    ref_feeder_gain_db: float  RT feeder gain of the reference satellite [dB].
    snr_thresh_db     : float  Sigmoid threshold [dB].  Defaults to
                               SNR_THRESH_DB from config when None.
    sigmoid_slope     : float  Sigmoid slope [1/dB].  Defaults to
                               SIGMOID_SLOPE from config when None.

    Returns
    -------
    float  Packet error rate in [0, 1).
    """
    if snr_thresh_db is None:
        snr_thresh_db = SNR_THRESH_DB
    if sigmoid_slope is None:
        sigmoid_slope = SIGMOID_SLOPE
    snr_db = _feeder_snr_db(cs, ref_feeder_gain_db)
    per    = 1.0 / (1.0 + math.exp(sigmoid_slope * (snr_db - snr_thresh_db)))
    return float(np.clip(per, 0.0, 0.99))


def _feeder_shannon_rate_mbps(cs: dict, ref_feeder_gain_db: float) -> float:
    """
    Shannon capacity [Mbps] of the feeder link.

    C = FEEDER_BANDWIDTH_HZ × log2(1 + SNR_linear)

    The result is clamped to a minimum of 1 Mbps so that NS-3 never
    receives a zero or negative data-rate string.

    Parameters
    ----------
    cs                : dict   Channel stats entry with feeder_* keys.
    ref_feeder_gain_db: float  RT feeder gain of the reference satellite [dB].

    Returns
    -------
    float  Capacity in Mbps (≥ 1.0).
    """
    snr_db  = _feeder_snr_db(cs, ref_feeder_gain_db)
    snr_lin = 10.0 ** (snr_db / 10.0)
    rate    = FEEDER_BANDWIDTH_HZ * math.log2(1.0 + snr_lin) / 1e6
    return max(rate, 1.0)


# =============================================================================
# Handover schedule computation
# =============================================================================

def _compute_handover_schedule(channel_stats: list,
                                tx_eirp_dbm: float = None,
                                ref_feeder_gain_db: float = None,
                                snr_thresh_db: float = None,
                                sigmoid_slope: float = None) -> list:
    """
    Determine the sequence of (satellite_id, start_time, end_time,
    per, delay_ms) intervals for the simulation duration.

    The algorithm:
    1. Sort satellites by decreasing elevation (highest = first to serve).
    2. Assign time slices weighted by 1/cos(elevation_rad): satellites near
       the horizon are visible for longer in a real orbital pass, so they
       receive proportionally more simulation time.
    3. Satellites below SAT_HANDOVER_ELEVATION_DEG are skipped (link drop).

    Parameters
    ----------
    channel_stats : list[dict]
        RT channel statistics, one dict per satellite (output of
        rt_sim.run_ray_tracing()).
    tx_eirp_dbm : float, optional
        Transmitter EIRP passed through to _rt_calibrated_per().
        Defaults to PHONE_EIRP_DBM when None.
    ref_feeder_gain_db : float, optional
        Reference (best-satellite) feeder path gain [dB] used for the
        feeder-link SNR correction.  Computed from channel_stats when None.
    snr_thresh_db : float, optional
        Sigmoid threshold [dB].  Defaults to SNR_THRESH_DB when None.
        Pass the value fitted to the Sionna LDPC BER curve.
    sigmoid_slope : float, optional
        Sigmoid slope [1/dB].  Defaults to SIGMOID_SLOPE when None.
        Pass the value fitted to the Sionna LDPC BER curve.

    Returns
    -------
    list of dicts with keys:
        sat_id           int     Satellite index.
        t_start          float   Start of service window [s].
        t_end            float   End of service window [s].
        elev_deg         float   Elevation angle [degrees].
        per              float   Service-link PER (RT-calibrated).
        delay_ms         float   Service-link one-way propagation delay [ms].
        feeder_per       float   Feeder-link PER (Ka-band RT-calibrated).
        feeder_rate_str  str     Feeder-link Shannon capacity as NS-3 string.
    """
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
    # This correctly gives more simulation time to low-elevation slots (near
    # the horizon, longer slant range, slower angular motion) and less time
    # to high-elevation slots (short slant range, rapid angular motion).
    #
    # Derivation cross-check (550 km, 60 s sim, sats at 70°/55°/40°):
    #   70° → ~14 s, 55° → ~18 s, 40° → ~28 s  (sum = 60 s)
    # The previous 1/cos(e) formula gave the opposite ranking (70° → 29 s)
    # because cos(e) is *largest* at low elevation, making 1/cos(e) *smallest*
    # there — physically backwards.
    #
    # Reference: orbital mechanics (Wertz, "Space Mission Engineering", §9.1;
    #            3GPP TR 38.821 §6.1 orbit geometry).
    elev_rads   = [math.radians(max(s["elevation_deg"], 1.0)) for s in visible]
    raw_weights = [
        _slant_range_m(SAT_HEIGHT_M, max(s["elevation_deg"], 1.0))
        / (SAT_ORBITAL_VELOCITY_MS * math.sin(e))
        for s, e in zip(visible, elev_rads)
    ]
    total_w     = sum(raw_weights)
    slot_durations = [w / total_w * SIM_DURATION_S for w in raw_weights]

    schedule = []

    # Reference gain: the best (highest-elevation) satellite's RT gain.
    # All other satellites are penalised relative to this value.
    ref_gain_db = visible[0]["mean_path_gain_db"] if visible else None

    # Reference feeder gain: best satellite's feeder path gain for Ka-band SNR.
    if ref_feeder_gain_db is None:
        ref_feeder_gain_db = max(
            (s["feeder_mean_path_gain_db"] for s in channel_stats
             if s.get("feeder_mean_path_gain_db", -200.0) > -150.0),
            default=-150.0,
        )

    t_cursor = 0.0
    for i, stat in enumerate(visible):
        slot_s   = slot_durations[i]
        t_start  = t_cursor
        t_end    = t_cursor + slot_s
        t_cursor = t_end
        elev_deg = stat["elevation_deg"]

        fspl   = _fspl_db(SAT_HEIGHT_M, max(elev_deg, 1.0))
        per    = _rt_calibrated_per(fspl, stat["mean_path_gain_db"],
                                    ref_gain_db, tx_eirp_dbm,
                                    snr_thresh_db, sigmoid_slope)
        delay  = _one_way_delay_ms(SAT_HEIGHT_M, max(elev_deg, 1.0))

        fdr_per      = _feeder_calibrated_per(stat, ref_feeder_gain_db,
                                              snr_thresh_db, sigmoid_slope)
        fdr_rate_mbps = _feeder_shannon_rate_mbps(stat, ref_feeder_gain_db)

        schedule.append(dict(
            sat_id          = stat["sat_id"],
            t_start         = round(t_start, 3),
            t_end           = round(t_end, 3),
            elev_deg        = elev_deg,
            per             = round(per, 4),
            delay_ms        = round(delay, 2),
            feeder_per      = round(fdr_per, 4),
            feeder_rate_str = f"{fdr_rate_mbps:.2f}Mbps",
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
       Δtput += (mean_RTT_s × link_rate_bps) / active_s   [converted to kbps]

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
       and preserves ssthresh.  TCP breaks the connection and requires a
       full slow-start, costing ~9 RTTs (TCP_RECOVERY_RTTS = 9) vs QUIC's
       1 RTT (QUIC_RECOVERY_RTTS = 1).  Source: RFC 9000 §9.3.
       Δtput = (TCP_RECOVERY_RTTS - QUIC_RECOVERY_RTTS) × mean_RTT_s
               × link_rate_bps / active_s   [per handover, summed]

    5. Loss reduction via PTO vs RTO
       QUIC does not collapse cwnd on loss, so effective loss ratio is lower.
        Reduction factor = exp(-0.8 × mean_RTT_s / REF_ONE_WAY_DELAY_S) × PER_ratio
        where REF_ONE_WAY_DELAY_S = 0.035 s (35 ms one-way reference delay,
        half of the 70 ms median LEO RTT reported in Sander et al. IMC 2022 and
        IETF draft-kuhn-quic-4-sat Table 1; used as the normalisation denominator).
       PER_ratio = mean(slot PER) over the schedule.

    6. HoL blocking elimination
       NOT applied — zero benefit for a single bulk stream.  HoL blocking
       elimination only matters for multi-stream / HTTP/3 workloads.

    Parameters
    ----------
    bbr_result    : dict   FlowMonitor result dict from a TCP BBR run.
    schedule      : list   Handover schedule (list of slot dicts with per, delay_ms).
    link_rate_bps : float  Actual service-link rate [bps].  When None, derived
                           from the schedule's feeder_rate_str values (first slot).
                           Passing the actual service-link rate here avoids
                           using an incorrect hardcoded value.

    Returns
    -------
    dict  QUIC result dict with corrected throughput_kbps, mean_delay_ms,
          loss_pct, and updated label/protocol fields.
    """
    import copy, re
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

    # Derive actual link rate from schedule if not provided.
    # Use the first slot's feeder_rate_str as a proxy for the service-link
    # rate (feeder ≥ service link by design, so this is an upper bound; in
    # practice the bottleneck is the service link at 10 Mbps hardcoded in
    # run_ns3).  Fall back to 10 Mbps if parsing fails.
    if link_rate_bps is None:
        try:
            rate_str  = schedule[0]["feeder_rate_str"] if schedule else "10.00Mbps"
            match     = re.match(r"([\d.]+)\s*([MKG]?)bps", rate_str, re.I)
            mult      = {"M": 1e6, "K": 1e3, "G": 1e9, "": 1.0}[match.group(2).upper()]
            link_rate_bps = float(match.group(1)) * mult
        except Exception:
            link_rate_bps = 10e6     # 10 Mbps service link (safe fallback)

    # RFC 9000 §9.3 post-handover recovery RTT counts
    TCP_RECOVERY_RTTS  = 9    # TCP slow-start + SYN from scratch
    QUIC_RECOVERY_RTTS = 1    # QUIC PATH_CHALLENGE/RESPONSE

    # Reference one-way delay for loss-reduction formula (Sander et al.
    # IMC 2022 — median LEO RTT = 70 ms → one-way = 35 ms; also consistent
    # with IETF draft-kuhn-quic-4-sat Table 1).  Used as the normalisation
    # denominator in the exp() reduction term; NOT the round-trip time.
    REF_ONE_WAY_DELAY_S = 0.035   # 35 ms one-way

    mss_bytes  = PACKET_SIZE_BYTES
    active_s   = SIM_DURATION_S - 1.0
    mean_rtt_s = max(bbr_result["mean_delay_ms"] / 1e3 * 2.0, 0.001)
    # NOTE: 2 × one-way delay is only a valid RTT estimate for symmetric paths.
    handovers  = bbr_result.get("handovers", 0)

    delta_kbps = 0.0

    # ── 1. 1-RTT handshake saving ─────────────────────────────────────────────
    # TCP wastes 2 RTTs before sending; QUIC wastes 1 RTT.
    # The extra RTT worth of pipe data translates to throughput credit.
    handshake_saving_kbps = (mean_rtt_s * link_rate_bps) / active_s / 1e3
    delta_kbps += handshake_saving_kbps

    # ── 2. PTO vs RTO: no cwnd collapse ───────────────────────────────────────
    # Estimate how many RTO events occurred (one per slot where PER exceeds the
    # fast-retransmit window threshold; approximated as PER > 0.03).
    # RTO rate ≈ PER / (mean_RTT_s × window_size_packets)
    # cwnd floor for PTO = 2 packets per RFC 9002 §6.2.4.
    cwnd_floor_pkts = 2   # RFC 9002 §6.2.4: minimum 2 packets during PTO probing
    rto_credit_kbps = 0.0
    for slot in schedule:
        slot_duration_s = slot["t_end"] - slot["t_start"]
        slot_per        = slot["per"]
        if slot_per > 0.03:
            window_pkts = max(int(link_rate_bps * mean_rtt_s / (mss_bytes * 8)), 4)
            rto_rate    = slot_per / (mean_rtt_s * window_pkts + 1e-9)
            slot_credit = cwnd_floor_pkts * mss_bytes * rto_rate * 8 / 1e3
            rto_credit_kbps += slot_credit * (slot_duration_s / active_s)
    delta_kbps += rto_credit_kbps

    # ── 3. Unlimited ACK ranges at high-PER slots ─────────────────────────────
    # At PER > 0.5, TCP SACK (3 block limit) wastes extra RTTs on ACK retries;
    # QUIC needs only 1 ACK.  Latency improvement ≈ mean_delay × (2/3) over
    # the high-PER fraction of the simulation.
    # Source: IETF draft-kuhn-quic-4-sat Table 1.
    high_per_fraction = sum(
        (s["t_end"] - s["t_start"]) / active_s
        for s in schedule if s["per"] > 0.5
    )
    latency_reduction_ms = bbr_result["mean_delay_ms"] * (2.0 / 3.0) * high_per_fraction

    # ── 4. Post-handover recovery ─────────────────────────────────────────────
    # QUIC PATH_CHALLENGE/RESPONSE costs 1 RTT vs TCP's ~9 RTT recovery.
    # Throughput credit per handover = RTT difference × link_rate / active_s.
    # Source: RFC 9000 §9.3.
    if handovers > 0:
        handover_credit_kbps = (
            (TCP_RECOVERY_RTTS - QUIC_RECOVERY_RTTS)
            * mean_rtt_s * link_rate_bps / active_s / 1e3
            * handovers
        )
        delta_kbps += handover_credit_kbps
    else:
        handover_credit_kbps = 0.0

    # ── 5. Loss reduction: PTO prevents cwnd collapse ─────────────────────────
    # QUIC PTO probes resolve isolated losses without RTO; effective loss
    # fraction decays exponentially with RTT relative to the reference RTT.
    # Source: IETF draft-kuhn-quic-4-sat Table 1; Sander et al. IMC 2022.
    mean_per = (sum(s["per"] * (s["t_end"] - s["t_start"])
                    for s in schedule) / active_s
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
# Main NS-3 simulation run
# =============================================================================

def run_ns3(scenario: str, protocol_cfg: dict, channel_stats: list,
            topology: str = "direct",
            snr_thresh_db: float = None,
            sigmoid_slope: float = None) -> dict:
    """
    Run one full NS-3 5G-NTN simulation with a moving satellite
    constellation for the specified transport protocol and topology.

    Parameters
    ----------
    scenario : str
        Channel scenario label (e.g. ``"urban"``).  Used for display only.
    protocol_cfg : dict
        Protocol configuration dict from config.PROTOCOLS, e.g.
        ``{"protocol": "tcp", "tcp_variant": "Cubic", "label": "TCP CUBIC"}``.
        For QUIC, the NS-3 simulation runs as UDP internally; RFC 9000
        corrections are applied after FlowMonitor collection.
    channel_stats : list[dict]
        RT channel statistics returned by rt_sim.run_ray_tracing().
        Used to calibrate per-satellite PER and slant-range delay.
    topology : str
        ``"direct"``  — Phone → Sat → GS → Server  (4 nodes, PHONE_EIRP_DBM)
        ``"indirect"`` — Phone → gNB → Sat → GS → Server  (5 nodes, GNB_EIRP_DBM)
    snr_thresh_db : float, optional
        Sigmoid threshold [dB] fitted to the Sionna LDPC BER curve.
        Defaults to SNR_THRESH_DB from config when None.
    sigmoid_slope : float, optional
        Sigmoid slope [1/dB] fitted to the Sionna LDPC BER curve.
        Defaults to SIGMOID_SLOPE from config when None.

    Returns
    -------
    dict with keys:
        scenario, protocol, label, topology, elevation_deg, svc_delay_ms,
        svc_loss_pct, tx_packets, rx_packets, loss_pct,
        mean_delay_ms, jitter_ms, throughput_kbps, handovers, schedule.
    """
    proto = protocol_cfg["protocol"]
    label = protocol_cfg.get("label", proto.upper())

    # QUIC runs as UDP in NS-3; corrections are applied post-simulation
    ns3_proto = "udp" if proto == "quic" else proto

    # Determine TX EIRP based on topology
    tx_eirp = GNB_EIRP_DBM if topology == "indirect" else PHONE_EIRP_DBM

    print(f"\n[NS-3]  {scenario.upper()}  topology={topology}  protocol={label}")

    # ── TCP global config ─────────────────────────────────────────────────────
    if ns3_proto == "tcp":
        _configure_tcp(protocol_cfg)

    # ── Feeder-link reference gain (needed before schedule is built) ──────────
    # Reference gain: best satellite's feeder path gain for Ka-band SNR norm.
    ref_feeder_gain = max(
        (s["feeder_mean_path_gain_db"] for s in channel_stats
         if s.get("feeder_mean_path_gain_db", -200.0) > -150.0),
        default=-150.0,
    )

    # ── Handover schedule derived from RT channel stats ───────────────────────
    # feeder_per and feeder_rate_str are now included in every schedule slot.
    schedule = _compute_handover_schedule(
        channel_stats, tx_eirp_dbm=tx_eirp,
        ref_feeder_gain_db=ref_feeder_gain,
        snr_thresh_db=snr_thresh_db,
        sigmoid_slope=sigmoid_slope,
    )
    print(f"  Handover schedule ({topology}, EIRP={tx_eirp:.0f} dBm): "
          f"{len(schedule)} slot(s)")
    for slot in schedule:
        print(f"    sat{slot['sat_id']}  {slot['t_start']:.1f}s–{slot['t_end']:.1f}s  "
              f"elev={slot['elev_deg']:.1f}°  "
              f"delay={slot['delay_ms']:.1f} ms  PER={slot['per']:.3f}  "
              f"fdr_PER={slot['feeder_per']:.3f}  fdr_rate={slot['feeder_rate_str']}")

    # Weighted-mean service-link propagation delay across all handover slots.
    # NS-3 PointToPointChannel.Delay is immutable after Install(), so we
    # must commit to a single delay value at topology creation time.
    # Using only the first slot's delay would underestimate the true average
    # because high-elevation slots (short delay) occupy less time than
    # near-horizon slots (long delay) in a real orbital pass.
    # The slot durations in the schedule are already weighted by 1/cos(elev),
    # so a simple duration-weighted mean is correct here.
    if schedule:
        total_dur = schedule[-1]["t_end"] - schedule[0]["t_start"]
        weighted_delay_ms = sum(
            s["delay_ms"] * (s["t_end"] - s["t_start"])
            for s in schedule
        ) / max(total_dur, 1e-9)
    else:
        weighted_delay_ms = _one_way_delay_ms(SAT_HEIGHT_M, 60.0)

    # Use the first (highest-elevation) slot for initial PER/rate parameters
    first = schedule[0] if schedule else dict(
        delay_ms=weighted_delay_ms,
        per=0.01,
        elev_deg=60.0,
        feeder_per=0.01,
        feeder_rate_str="100.00Mbps",
    )

    # ── Feeder-link initial values (from first slot) ──────────────────────────
    # Channel delay is fixed at topology creation (NS-3 P2P channel immutable).
    # DataRate and PER can be updated at runtime on each handover.
    active_sat_id = int(first.get("sat_id", 0)) if schedule else 0
    active_cs     = (channel_stats[active_sat_id]
                     if channel_stats and active_sat_id < len(channel_stats) else {})
    fdr_delay_ms  = float(active_cs.get("feeder_propagation_delay_ms",
                                         _one_way_delay_ms(SAT_HEIGHT_M, 60.0)))
    fdr_rate_str  = first.get("feeder_rate_str", "100.00Mbps")
    fdr_per       = first.get("feeder_per", 0.01)
    print(f"  Service link delay (weighted-mean): {weighted_delay_ms:.2f} ms")
    print(f"  Feeder link (sat→GS, Ka-band):  "
          f"delay={fdr_delay_ms:.2f} ms  "
          f"rate={fdr_rate_str}  "
          f"PER={fdr_per:.4f}  "
          f"(ref_gain={ref_feeder_gain:.1f} dB)")

    # ── Build node topology ───────────────────────────────────────────────────
    import collections

    if topology == "indirect":
        # 5 nodes: Phone | gNB | Satellite | GroundStation | Server
        nodes = ns.NodeContainer()
        nodes.Create(5)
        phone, gnb, sat, gs, server = (nodes.Get(i) for i in range(5))
        ns.InternetStackHelper().Install(nodes)

        # Phone ↔ gNB (terrestrial radio hop)
        p2p = ns.PointToPointHelper()
        p2p.SetDeviceAttribute("DataRate", ns.StringValue(GNB_DATARATE))
        p2p.SetChannelAttribute("Delay",
                                ns.StringValue(f"{GNB_PROCESSING_DELAY_MS:.3f}ms"))
        devs_ue_gnb = p2p.Install(_nc2(phone, gnb))
        # Terrestrial PER on the Phone→gNB hop
        em_terr = ns.CreateObject[ns.RateErrorModel]()
        em_terr.SetAttribute("ErrorRate", ns.DoubleValue(GNB_TERRESTRIAL_PER))
        em_terr.SetAttribute("ErrorUnit", ns.StringValue("ERROR_UNIT_PACKET"))
        devs_ue_gnb.Get(1).SetAttribute("ReceiveErrorModel",
                                         ns.PointerValue(em_terr))

        # gNB ↔ Satellite (NTN hop, GNB_EIRP)
        # Channel delay is fixed at topology creation (NS-3 P2P immutable).
        # Use weighted-mean delay across all handover slots so the fixed
        # delay accurately represents the full simulation period, not just
        # the first (highest-elevation) slot.
        p2p.SetDeviceAttribute("DataRate", ns.StringValue("10Mbps"))
        p2p.SetChannelAttribute("Delay",
                                ns.StringValue(f"{weighted_delay_ms:.3f}ms"))
        devs_svc = p2p.Install(_nc2(gnb, sat))
        em = ns.CreateObject[ns.RateErrorModel]()
        em.SetAttribute("ErrorRate", ns.DoubleValue(first["per"]))
        em.SetAttribute("ErrorUnit", ns.StringValue("ERROR_UNIT_PACKET"))
        devs_svc.Get(1).SetAttribute("ReceiveErrorModel", ns.PointerValue(em))

        # Satellite ↔ GS (feeder link — RT-derived Ka-band)
        p2p.SetDeviceAttribute("DataRate",  ns.StringValue(fdr_rate_str))
        p2p.SetChannelAttribute("Delay",    ns.StringValue(f"{fdr_delay_ms:.3f}ms"))
        devs_fdr = p2p.Install(_nc2(sat, gs))
        em_fdr = ns.CreateObject[ns.RateErrorModel]()
        em_fdr.SetAttribute("ErrorRate", ns.DoubleValue(fdr_per))
        em_fdr.SetAttribute("ErrorUnit", ns.StringValue("ERROR_UNIT_PACKET"))
        devs_fdr.Get(1).SetAttribute("ReceiveErrorModel", ns.PointerValue(em_fdr))

        # GS ↔ Server
        p2p.SetDeviceAttribute("DataRate",  ns.StringValue("1Gbps"))
        p2p.SetChannelAttribute("Delay",    ns.StringValue(f"{TERRESTRIAL_BACKHAUL_DELAY_MS:.0f}ms"))
        devs_gnd = p2p.Install(_nc2(gs, server))

        # IP addressing
        ipv4 = ns.Ipv4AddressHelper()
        ipv4.SetBase(ns.Ipv4Address("10.1.0.0"), ns.Ipv4Mask("255.255.255.0"))
        ipv4.Assign(devs_ue_gnb)
        ipv4.SetBase(ns.Ipv4Address("10.1.1.0"), ns.Ipv4Mask("255.255.255.0"))
        iface_svc = ipv4.Assign(devs_svc)
        ipv4.SetBase(ns.Ipv4Address("10.1.2.0"), ns.Ipv4Mask("255.255.255.0"))
        ipv4.Assign(devs_fdr)
        ipv4.SetBase(ns.Ipv4Address("10.1.3.0"), ns.Ipv4Mask("255.255.255.0"))
        iface_gnd = ipv4.Assign(devs_gnd)

    else:
        # 4 nodes: Phone | Satellite | GroundStation | Server  (direct)
        nodes = ns.NodeContainer()
        nodes.Create(4)
        phone, sat, gs, server = (nodes.Get(i) for i in range(4))
        ns.InternetStackHelper().Install(nodes)

        # Service link: Phone ↔ Satellite
        # Channel delay is fixed at topology creation (NS-3 P2P immutable).
        # Use weighted-mean delay across all handover slots.
        p2p = ns.PointToPointHelper()
        p2p.SetDeviceAttribute("DataRate", ns.StringValue("10Mbps"))
        p2p.SetChannelAttribute("Delay",
                                ns.StringValue(f"{weighted_delay_ms:.3f}ms"))
        devs_svc = p2p.Install(_nc2(phone, sat))
        em = ns.CreateObject[ns.RateErrorModel]()
        em.SetAttribute("ErrorRate", ns.DoubleValue(first["per"]))
        em.SetAttribute("ErrorUnit", ns.StringValue("ERROR_UNIT_PACKET"))
        devs_svc.Get(1).SetAttribute("ReceiveErrorModel", ns.PointerValue(em))

        # Feeder link: Satellite ↔ GS (RT-derived Ka-band)
        p2p.SetDeviceAttribute("DataRate",  ns.StringValue(fdr_rate_str))
        p2p.SetChannelAttribute("Delay",    ns.StringValue(f"{fdr_delay_ms:.3f}ms"))
        devs_fdr = p2p.Install(_nc2(sat, gs))
        em_fdr = ns.CreateObject[ns.RateErrorModel]()
        em_fdr.SetAttribute("ErrorRate", ns.DoubleValue(fdr_per))
        em_fdr.SetAttribute("ErrorUnit", ns.StringValue("ERROR_UNIT_PACKET"))
        devs_fdr.Get(1).SetAttribute("ReceiveErrorModel", ns.PointerValue(em_fdr))

        # Terrestrial: GS ↔ Server
        p2p.SetDeviceAttribute("DataRate",  ns.StringValue("1Gbps"))
        p2p.SetChannelAttribute("Delay",    ns.StringValue(f"{TERRESTRIAL_BACKHAUL_DELAY_MS:.0f}ms"))
        devs_gnd = p2p.Install(_nc2(gs, server))

        # IP addressing
        ipv4 = ns.Ipv4AddressHelper()
        ipv4.SetBase(ns.Ipv4Address("10.1.1.0"), ns.Ipv4Mask("255.255.255.0"))
        iface_svc = ipv4.Assign(devs_svc)
        ipv4.SetBase(ns.Ipv4Address("10.1.2.0"), ns.Ipv4Mask("255.255.255.0"))
        ipv4.Assign(devs_fdr)
        ipv4.SetBase(ns.Ipv4Address("10.1.3.0"), ns.Ipv4Mask("255.255.255.0"))
        iface_gnd = ipv4.Assign(devs_gnd)

    ns.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

    # ── Satellite mobility: LEO orbit at ~7.6 km/s ────────────────────────────
    static_mob = ns.MobilityHelper()
    static_mob.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    static_mob.Install(nodes)

    sat_mob = ns.CreateObject[ns.ConstantVelocityMobilityModel]()
    sat.AggregateObject(sat_mob)
    sat_mob.SetPosition(ns.Vector(0.0, 0.0, SAT_HEIGHT_M))
    sat_mob.SetVelocity(ns.Vector(SAT_ORBITAL_VELOCITY_MS, 0.0, 0.0))

    # ── Schedule handover events ──────────────────────────────────────────────
    # cppyy converts Python callables to C function pointers void(*)().
    # All distinct Python closures map to the *same* C function pointer, so
    # we cannot pass different closures for different handover times.
    # Instead, we drive handovers with a single function that pops the next
    # (time, per, slot) entry from a module-level deque on each call, then
    # reschedules itself for the following handover.

    handover_queue = collections.deque(schedule[1:])   # remaining handovers
    handover_count = [0]

    def _handover_tick():
        """
        Single handover callback — applies the next PER/rate from the queue
        and reschedules itself for the subsequent handover time (if any).

        Updates both the service-link error model (em) and the feeder-link
        error model (em_fdr) plus feeder device DataRate on each handover.
        The feeder-link channel delay cannot be updated at runtime (NS-3
        PointToPointChannel.Delay is immutable after Install).
        """
        if not handover_queue:
            return
        slot = handover_queue.popleft()
        # Service-link PER
        em.SetAttribute("ErrorRate", ns.DoubleValue(slot["per"]))
        # Feeder-link PER and DataRate (delay is fixed at topology creation)
        em_fdr.SetAttribute("ErrorRate", ns.DoubleValue(slot["feeder_per"]))
        devs_fdr.Get(0).SetAttribute("DataRate",
                                     ns.StringValue(slot["feeder_rate_str"]))
        handover_count[0] += 1
        print(f"  [t={ns.Simulator.Now().GetSeconds():.1f}s] "
              f"Handover → sat{slot['sat_id']}  "
              f"PER={slot['per']:.3f}  elev={slot['elev_deg']:.1f}°  "
              f"fdr_PER={slot['feeder_per']:.3f}  "
              f"fdr_rate={slot['feeder_rate_str']}")
        # Schedule next handover
        if handover_queue:
            nxt = handover_queue[0]
            if nxt["t_start"] < SIM_DURATION_S:
                ev = ns.cppyy.gbl.pythonMakeEvent(_handover_tick)
                ns.Simulator.Schedule(ns.Seconds(nxt["t_start"]), ev)

    # Kick off the first handover
    if len(schedule) > 1:
        first_ho = schedule[1]
        if first_ho["t_start"] < SIM_DURATION_S:
            ev = ns.cppyy.gbl.pythonMakeEvent(_handover_tick)
            ns.Simulator.Schedule(ns.Seconds(first_ho["t_start"]), ev)

    # ── Application layer ─────────────────────────────────────────────────────
    port = 9
    server_addr = ns.InetSocketAddress(iface_gnd.GetAddress(1), port)

    if ns3_proto == "udp":
        # Constant-bit-rate UDP (OnOff)
        onoff = ns.OnOffHelper(
            "ns3::UdpSocketFactory", server_addr.ConvertTo())
        onoff.SetAttribute("DataRate",   ns.StringValue(APP_DATA_RATE))
        onoff.SetAttribute("PacketSize", ns.UintegerValue(PACKET_SIZE_BYTES))
        onoff.SetAttribute("OnTime",
            ns.StringValue("ns3::ConstantRandomVariable[Constant=1]"))
        onoff.SetAttribute("OffTime",
            ns.StringValue("ns3::ConstantRandomVariable[Constant=0]"))
        for _ in range(NUM_PARALLEL_FLOWS):
            app_tx = onoff.Install(phone)
            app_tx.Start(ns.Seconds(1.0))
            app_tx.Stop(ns.Seconds(SIM_DURATION_S))

        sink = ns.PacketSinkHelper(
            "ns3::UdpSocketFactory",
            ns.InetSocketAddress(ns.Ipv4Address.GetAny(), port).ConvertTo())
        ip_proto_num = 17

    else:
        # TCP: saturating BulkSend (fills the pipe)
        bulk = ns.BulkSendHelper(
            "ns3::TcpSocketFactory", server_addr.ConvertTo())
        bulk.SetAttribute("SendSize", ns.UintegerValue(PACKET_SIZE_BYTES))
        bulk.SetAttribute("MaxBytes", ns.UintegerValue(0))   # unlimited
        for _ in range(NUM_PARALLEL_FLOWS):
            app_tx = bulk.Install(phone)
            app_tx.Start(ns.Seconds(1.0))
            app_tx.Stop(ns.Seconds(SIM_DURATION_S))

        sink = ns.PacketSinkHelper(
            "ns3::TcpSocketFactory",
            ns.InetSocketAddress(ns.Ipv4Address.GetAny(), port).ConvertTo())
        ip_proto_num = 6

    app_rx = sink.Install(server)
    app_rx.Start(ns.Seconds(0.0))
    app_rx.Stop(ns.Seconds(SIM_DURATION_S + 2.0))

    # ── FlowMonitor ───────────────────────────────────────────────────────────
    fm_helper = ns.FlowMonitorHelper()
    monitor   = fm_helper.InstallAll()

    # ── Run simulation ────────────────────────────────────────────────────────
    ns.Simulator.Stop(ns.Seconds(SIM_DURATION_S + 3.0))
    ns.Simulator.Run()
    monitor.CheckForLostPackets()

    # ── Collect FlowMonitor statistics ────────────────────────────────────────
    stats      = monitor.GetFlowStats()
    classifier = fm_helper.GetClassifier()

    result = dict(
        scenario        = scenario,
        protocol        = proto,       # original protocol name (not ns3_proto)
        label           = label,
        topology        = topology,
        elevation_deg   = first["elev_deg"],
        svc_delay_ms    = round(weighted_delay_ms, 2),   # weighted-mean, not first-slot only
        svc_loss_pct    = round(first["per"] * 100.0, 2),
        tx_packets      = 0,
        rx_packets      = 0,
        loss_pct        = 0.0,
        mean_delay_ms   = 0.0,
        jitter_ms       = 0.0,
        throughput_kbps = 0.0,
        handovers       = handover_count[0],
        schedule        = schedule,
    )

    for pair in stats:
        fid, fs = pair.first, pair.second
        if classifier.FindFlow(fid).protocol != ip_proto_num:
            continue
        if fs.rxPackets == 0:
            continue
        rx_n = int(fs.rxPackets)
        tx_n = int(fs.txPackets)
        active_s = SIM_DURATION_S - 1.0   # Exclude 1 s warm-up
        result.update(
            tx_packets      = result["tx_packets"] + tx_n,
            rx_packets      = result["rx_packets"] + rx_n,
            loss_pct        = round(100.0 * (1 - rx_n / max(tx_n, 1)), 2),
            mean_delay_ms   = round(
                fs.delaySum.GetSeconds() / rx_n * 1e3, 2),
            jitter_ms       = round(
                fs.jitterSum.GetSeconds() / max(rx_n - 1, 1) * 1e3, 2),
            throughput_kbps = result["throughput_kbps"] + round(
                fs.rxBytes * 8.0 / active_s / 1e3, 2),
        )

    ns.Simulator.Destroy()

    print(f"\n  Results ({label}, {topology}):")
    for k, v in result.items():
        if k != "schedule":
            print(f"    {k:<24s}: {v}")

    return result


# =============================================================================
# Protocol suite runner — both topologies
# =============================================================================

def run_ns3_both_topologies(channel_stats: list,
                             scenario: str = "urban",
                             snr_thresh_db: float = None,
                             sigmoid_slope: float = None) -> tuple:
    """
    Run the NS-3 simulation once per entry in config.PROTOCOLS × 2
    topologies (direct / indirect) and return all results for plotting.

    For QUIC, the NS-3 simulation runs as UDP internally.  After the
    BBR run completes, RFC 9000 corrections are applied analytically to
    produce the QUIC result (no additional NS-3 simulation needed).

    Parameters
    ----------
    channel_stats : list[dict]
        RT channel statistics from rt_sim.run_ray_tracing().
    scenario : str
        NTN propagation scenario label (used for display only).
    snr_thresh_db : float, optional
        Sigmoid threshold [dB] fitted to the Sionna LDPC BER curve.
        Defaults to SNR_THRESH_DB from config when None.
    sigmoid_slope : float, optional
        Sigmoid slope [1/dB] fitted to the Sionna LDPC BER curve.
        Defaults to SIGMOID_SLOPE from config when None.

    Returns
    -------
    tuple (direct_results, indirect_results)
        Each is a list[dict], one result dict per protocol.
    """
    direct_results   = []
    indirect_results = []

    for topology, results_list in [("direct",   direct_results),
                                    ("indirect", indirect_results)]:
        bbr_result_for_quic = None

        for pcfg in PROTOCOLS:
            if pcfg["protocol"] == "quic":
                # Defer QUIC until after BBR has run
                continue

            full_cfg = dict(
                tcp_snd_buf    = TCP_SNDRCV_BUF_BYTES,
                tcp_rcv_buf    = TCP_SNDRCV_BUF_BYTES,
                tcp_sack       = TCP_SACK_ENABLED,
                tcp_timestamps = TCP_TIMESTAMPS,
                **pcfg,
            )
            r = run_ns3(scenario, full_cfg, channel_stats, topology=topology,
                        snr_thresh_db=snr_thresh_db, sigmoid_slope=sigmoid_slope)
            results_list.append(r)

            if pcfg.get("label") == "TCP BBR":
                bbr_result_for_quic = r

        # Now compute QUIC from BBR baseline.
        # Pass the actual service-link rate (10 Mbps hardcoded in run_ns3)
        # so _apply_quic_corrections does not fall back to parsing feeder rate.
        if bbr_result_for_quic is not None:
            # Use the schedule from the BBR run (same topology/EIRP)
            quic_result = _apply_quic_corrections(
                bbr_result_for_quic, bbr_result_for_quic["schedule"],
                link_rate_bps=10e6)   # 10 Mbps service link (NS-3 DataRate)
            quic_result["topology"] = topology
            results_list.append(quic_result)
        else:
            print(f"[QUIC]  No BBR result found for {topology} topology — "
                  "skipping QUIC.")

    return direct_results, indirect_results


# =============================================================================
# Legacy single-topology runner (backward compatibility)
# =============================================================================

def run_ns3_protocol_suite(channel_stats: list,
                            scenario: str = "urban") -> list:
    """
    Run the NS-3 simulation once per entry in config.PROTOCOLS for the
    direct topology only.  Returns a flat list of results.

    Kept for backward compatibility with older callers.  New code should
    use run_ns3_both_topologies() instead.
    """
    direct_results, _ = run_ns3_both_topologies(channel_stats, scenario)
    return direct_results
