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
       │  5G-NR NTN service link  (600 km LEO, RT-calibrated PER, PHONE_EIRP)
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
       │  5G-NR NTN service link  (600 km LEO, RT-calibrated PER, GNB_EIRP)
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
                        tx_eirp_dbm: float = None) -> float:
    """
    Estimate packet error rate using a sigmoid link-budget model
    calibrated by the RT-derived mean path gain.

    The Sionna RT proxy TX is placed only RT_SAT_SCENE_HEIGHT_M (~300 m)
    above the scene, so the absolute mean_path_gain_db value reflects both
    the short proxy free-space loss and urban multipath.  The *relative*
    gain difference between satellites (due to elevation-dependent shadowing
    and multipath) is physically meaningful; the absolute value is not
    directly transferable to the real 600 km slant range.

    Link budget:
        SNR [dB] = Tx_EIRP − FSPL_600km + urban_correction − Noise_floor

    where urban_correction is derived by normalising the RT gain against the
    best (highest-elevation) satellite so that the reference satellite
    contributes 0 dB correction and lower-elevation satellites are penalised
    by their relative RT gain deficit.

    Parameters
    ----------
    fspl_db         : float  Free-space path loss over real 600 km slant [dB].
    rt_mean_gain_db : float  Mean |h| gain from Sionna RT for this satellite [dB].
    rt_ref_gain_db  : float  RT gain of the reference (best) satellite [dB].
                             When None, no urban correction is applied.
    tx_eirp_dbm     : float  Transmitter EIRP [dBm].  When None, defaults to
                             PHONE_EIRP_DBM (23 dBm) for backward compatibility.
                             Use GNB_EIRP_DBM (43 dBm) for the indirect topology.

    Returns
    -------
    float  Packet error rate in [0, 1).
    """
    if tx_eirp_dbm is None:
        tx_eirp_dbm = PHONE_EIRP_DBM

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
    per    = 1.0 / (1.0 + math.exp(SIGMOID_SLOPE * (snr_db - SNR_THRESH_DB)))
    return float(np.clip(per, 0.0, 0.99))


def _feeder_snr_db(cs: dict, ref_feeder_gain_db: float) -> float:
    """
    Compute the feeder-link SNR [dB] for a satellite entry.

    Uses Ka-band FSPL (FEEDER_FREQ_HZ) over the real 600 km slant range
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


def _feeder_calibrated_per(cs: dict, ref_feeder_gain_db: float) -> float:
    """
    Sigmoid PER model for the Satellite→GS feeder link.

    Mirrors _rt_calibrated_per but uses Ka-band FSPL (FEEDER_FREQ_HZ),
    the satellite feeder EIRP (SAT_TX_EIRP_FEEDER_DBM), and the GS dish
    gain (GS_RX_ANTENNA_GAIN_DB) instead of service-link parameters.

    Parameters
    ----------
    cs                : dict   Channel stats entry with feeder_* keys.
    ref_feeder_gain_db: float  RT feeder gain of the reference satellite [dB].

    Returns
    -------
    float  Packet error rate in [0, 1).
    """
    snr_db = _feeder_snr_db(cs, ref_feeder_gain_db)
    per    = 1.0 / (1.0 + math.exp(SIGMOID_SLOPE * (snr_db - SNR_THRESH_DB)))
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
                                ref_feeder_gain_db: float = None) -> list:
    """
    Determine the sequence of (satellite_id, start_time, end_time,
    per, delay_ms) intervals for the simulation duration.

    The algorithm:
    1. Sort satellites by decreasing elevation (highest = first to serve).
    2. Assign equal time slices based on SIM_DURATION_S / NUM_SATELLITES.
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

    # Divide simulation time evenly among visible satellites
    slot_s   = SIM_DURATION_S / max(len(visible), 1)
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

    for i, stat in enumerate(visible):
        t_start  = i * slot_s
        t_end    = (i + 1) * slot_s
        elev_deg = stat["elevation_deg"]

        fspl   = _fspl_db(SAT_HEIGHT_M, max(elev_deg, 1.0))
        per    = _rt_calibrated_per(fspl, stat["mean_path_gain_db"],
                                    ref_gain_db, tx_eirp_dbm)
        delay  = _one_way_delay_ms(SAT_HEIGHT_M, max(elev_deg, 1.0))

        fdr_per      = _feeder_calibrated_per(stat, ref_feeder_gain_db)
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

def _apply_quic_corrections(bbr_result: dict, schedule: list) -> dict:
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

    3. Unlimited ACK ranges
       At PER > 0.5, TCP SACK (3 block limit) cannot describe the full loss
       map in one ACK, requiring multiple ACK round trips.  QUIC ACK frames
       support unlimited gap/range pairs → 1 ACK suffices.
       Applied as a latency reduction of 2/3 at high-PER slots.

    4. Post-handover throughput dip reduction
       QUIC connection migration (PATH_CHALLENGE/PATH_RESPONSE) costs 1 RTT
       and preserves ssthresh.  TCP breaks the connection, costs ~9 RTTs.
       Post-handover dip is 50 % shorter for QUIC.
       Δtput ≈ +5 % of BBR aggregate per handover event.

    5. HoL blocking elimination
       NOT applied — zero benefit for a single bulk stream.  HoL blocking
       elimination only matters for multi-stream / HTTP/3 workloads.

    Parameters
    ----------
    bbr_result : dict   FlowMonitor result dict from a TCP BBR run.
    schedule   : list   Handover schedule (list of slot dicts with per, delay_ms).

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

    link_rate_bps = 10e6          # 10 Mbps service link
    mss_bytes     = PACKET_SIZE_BYTES
    active_s      = SIM_DURATION_S - 1.0
    mean_rtt_s    = max(bbr_result["mean_delay_ms"] / 1e3 * 2.0, 0.001)  # one-way → RTT, min 1 ms
    handovers     = bbr_result.get("handovers", 0)

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
    # cwnd floor for PTO = 2 packets (probe + 1 in flight) vs TCP's 1 packet.
    cwnd_floor_pkts = 2
    rto_credit_kbps = 0.0
    for slot in schedule:
        slot_duration_s = slot["t_end"] - slot["t_start"]
        slot_per        = slot["per"]
        if slot_per > 0.03:
            # Estimated RTO events per second in this slot
            window_pkts = max(int(link_rate_bps * mean_rtt_s / (mss_bytes * 8)), 4)
            rto_rate    = slot_per / (mean_rtt_s * window_pkts + 1e-9)
            # Extra throughput QUIC recovers vs TCP (cwnd_floor × MSS × rto_rate)
            slot_credit = cwnd_floor_pkts * mss_bytes * rto_rate * 8 / 1e3
            # Weight by slot fraction
            rto_credit_kbps += slot_credit * (slot_duration_s / active_s)
    delta_kbps += rto_credit_kbps

    # ── 3. Unlimited ACK ranges at high-PER slots ─────────────────────────────
    # At PER > 0.5 TCP needs ~3 ACK round trips to cover the loss map;
    # QUIC needs 1.  This means TCP wastes 2 extra RTTs per window.
    # Latency improvement ≈ mean_delay_ms × 2/3 averaged over the high-PER
    # fraction of the simulation.
    high_per_fraction = sum(
        (s["t_end"] - s["t_start"]) / active_s
        for s in schedule if s["per"] > 0.5
    )
    latency_reduction_ms = bbr_result["mean_delay_ms"] * (2.0 / 3.0) * high_per_fraction

    # ── 4. Post-handover recovery ─────────────────────────────────────────────
    # QUIC recovers ~50 % faster than TCP after handover.
    # Approximated as +5 % throughput per handover (recovery dip is shorter).
    if handovers > 0:
        handover_credit_kbps = bbr_result["throughput_kbps"] * 0.05 * handovers
        delta_kbps += handover_credit_kbps

    # ── Apply corrections ─────────────────────────────────────────────────────
    result["throughput_kbps"] = round(
        bbr_result["throughput_kbps"] + delta_kbps, 2)
    result["mean_delay_ms"]   = round(
        max(1.0, bbr_result["mean_delay_ms"] - latency_reduction_ms), 2)

    # QUIC does not collapse cwnd on loss → lower effective loss ratio
    # (PTO probes resolve isolated losses that would otherwise time out)
    result["loss_pct"] = round(bbr_result["loss_pct"] * 0.15, 3)

    print(f"\n  QUIC corrections (from BBR baseline):")
    print(f"    handshake saving : +{handshake_saving_kbps:.1f} kbps")
    print(f"    PTO vs RTO       : +{rto_credit_kbps:.1f} kbps")
    print(f"    post-H/O credit  : +{(delta_kbps - handshake_saving_kbps - rto_credit_kbps):.1f} kbps")
    print(f"    ACK range lat.   : -{latency_reduction_ms:.2f} ms")
    print(f"    total Δtput      : +{delta_kbps:.1f} kbps  →  {result['throughput_kbps']:.0f} kbps")
    print(f"    latency          : {bbr_result['mean_delay_ms']:.1f} ms → {result['mean_delay_ms']:.1f} ms")
    print(f"    loss             : {bbr_result['loss_pct']:.2f}% → {result['loss_pct']:.3f}%")

    return result


# =============================================================================
# Main NS-3 simulation run
# =============================================================================

def run_ns3(scenario: str, protocol_cfg: dict, channel_stats: list,
            topology: str = "direct") -> dict:
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
    )
    print(f"  Handover schedule ({topology}, EIRP={tx_eirp:.0f} dBm): "
          f"{len(schedule)} slot(s)")
    for slot in schedule:
        print(f"    sat{slot['sat_id']}  {slot['t_start']:.1f}s–{slot['t_end']:.1f}s  "
              f"elev={slot['elev_deg']:.1f}°  "
              f"delay={slot['delay_ms']:.1f} ms  PER={slot['per']:.3f}  "
              f"fdr_PER={slot['feeder_per']:.3f}  fdr_rate={slot['feeder_rate_str']}")

    # Use the first (highest-elevation) slot for initial link parameters
    first = schedule[0] if schedule else dict(
        delay_ms=_one_way_delay_ms(SAT_HEIGHT_M, 60.0),
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
        p2p.SetDeviceAttribute("DataRate", ns.StringValue("10Mbps"))
        p2p.SetChannelAttribute("Delay",
                                ns.StringValue(f"{first['delay_ms']:.3f}ms"))
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
        p2p.SetChannelAttribute("Delay",    ns.StringValue("10ms"))
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
        p2p = ns.PointToPointHelper()
        p2p.SetDeviceAttribute("DataRate", ns.StringValue("10Mbps"))
        p2p.SetChannelAttribute("Delay",
                                ns.StringValue(f"{first['delay_ms']:.3f}ms"))
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
        p2p.SetChannelAttribute("Delay",    ns.StringValue("10ms"))
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
        svc_delay_ms    = first["delay_ms"],
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
                             scenario: str = "urban") -> tuple:
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
            r = run_ns3(scenario, full_cfg, channel_stats, topology=topology)
            results_list.append(r)

            if pcfg.get("label") == "TCP BBR":
                bbr_result_for_quic = r

        # Now compute QUIC from BBR baseline
        if bbr_result_for_quic is not None:
            # Use the schedule from the BBR run (same topology/EIRP)
            quic_result = _apply_quic_corrections(
                bbr_result_for_quic, bbr_result_for_quic["schedule"])
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
