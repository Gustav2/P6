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
  * Multi-protocol comparison — the function run_ns3_protocol_suite()
    runs the full simulation once per entry in config.PROTOCOLS and
    returns all results for plotting.

Topology
--------

  Phone (UE)
     │  5G-NR NTN service link  (600 km LEO, RT-calibrated PER)
     ▼
  Satellite [moving]     ← handover when elevation < SAT_HANDOVER_ELEVATION_DEG
     │  Ka-band feeder link  (100 Mbps, ~2 ms)
     ▼
  Ground Station
     │  Terrestrial fibre  (1 Gbps, 10 ms)
     ▼
  Internet Server

Satellite mobility
------------------
Each satellite node is given a ConstantVelocity mobility model.
NS-3 does not enforce signal interruption based on position, so the
handover is modelled by:
  1. Scheduling a link-rate change callback at the handover time.
  2. Changing the error model on the service link to reflect the new
     satellite's RT-calibrated PER at its elevation angle.

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
                        rt_ref_gain_db: float = None) -> float:
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

    Returns
    -------
    float  Packet error rate in [0, 1).
    """
    TX_EIRP_DBM     = 50.0    # Satellite beam EIRP [dBm] (~100 W, realistic LEO NTN)
    NOISE_FLOOR_DBM = -120.0  # Thermal noise at UE [dBm]
    SNR_THRESH_DB   =  7.5    # QPSK r=0.5 link threshold [dB]
    SIGMOID_SLOPE   =  0.7    # Slope of the PER sigmoid curve

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

    snr_db = TX_EIRP_DBM - fspl_db + urban_correction_db - NOISE_FLOOR_DBM
    per    = 1.0 / (1.0 + math.exp(SIGMOID_SLOPE * (snr_db - SNR_THRESH_DB)))
    return float(np.clip(per, 0.0, 0.99))


# =============================================================================
# Handover schedule computation
# =============================================================================

def _compute_handover_schedule(channel_stats: list) -> list:
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

    Returns
    -------
    list of dicts with keys:
        sat_id      int     Satellite index.
        t_start     float   Start of service window [s].
        t_end       float   End of service window [s].
        elev_deg    float   Elevation angle [degrees].
        per         float   Packet error rate (RT-calibrated).
        delay_ms    float   One-way propagation delay [ms].
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

    for i, stat in enumerate(visible):
        t_start  = i * slot_s
        t_end    = (i + 1) * slot_s
        elev_deg = stat["elevation_deg"]

        fspl   = _fspl_db(SAT_HEIGHT_M, max(elev_deg, 1.0))
        per    = _rt_calibrated_per(fspl, stat["mean_path_gain_db"], ref_gain_db)
        delay  = _one_way_delay_ms(SAT_HEIGHT_M, max(elev_deg, 1.0))

        schedule.append(dict(
            sat_id    = stat["sat_id"],
            t_start   = round(t_start, 3),
            t_end     = round(t_end, 3),
            elev_deg  = elev_deg,
            per       = round(per, 4),
            delay_ms  = round(delay, 2),
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
# Main NS-3 simulation run
# =============================================================================

def run_ns3(scenario: str, protocol_cfg: dict, channel_stats: list) -> dict:
    """
    Run one full NS-3 5G-NTN simulation with a moving satellite
    constellation for the specified transport protocol.

    Parameters
    ----------
    scenario : str
        Channel scenario label (e.g. ``"urban"``).  Used for display only.
    protocol_cfg : dict
        Protocol configuration dict from config.PROTOCOLS, e.g.
        ``{"protocol": "tcp", "tcp_variant": "Cubic", "label": "TCP CUBIC"}``.
    channel_stats : list[dict]
        RT channel statistics returned by rt_sim.run_ray_tracing().
        Used to calibrate per-satellite PER and slant-range delay.

    Returns
    -------
    dict with keys:
        scenario, protocol, label, elevation_deg, svc_delay_ms,
        svc_loss_pct, tx_packets, rx_packets, loss_pct,
        mean_delay_ms, jitter_ms, throughput_kbps, handovers.
    """
    proto = protocol_cfg["protocol"]
    label = protocol_cfg.get("label", proto.upper())

    print(f"\n[NS-3]  {scenario.upper()}  protocol={label}")

    # ── TCP global config ─────────────────────────────────────────────────────
    if proto == "tcp":
        _configure_tcp(protocol_cfg)

    # ── Handover schedule derived from RT channel stats ───────────────────────
    schedule = _compute_handover_schedule(channel_stats)
    print(f"  Handover schedule: {len(schedule)} slot(s)")
    for slot in schedule:
        print(f"    sat{slot['sat_id']}  {slot['t_start']:.1f}s–{slot['t_end']:.1f}s  "
              f"elev={slot['elev_deg']:.1f}°  "
              f"delay={slot['delay_ms']:.1f} ms  PER={slot['per']:.3f}")

    # Use the first (highest-elevation) slot for initial link parameters
    first = schedule[0] if schedule else dict(
        delay_ms=_one_way_delay_ms(SAT_HEIGHT_M, 60.0),
        per=0.01,
        elev_deg=60.0,
    )

    # ── 4 nodes: Phone | Satellite | GroundStation | Server ──────────────────
    nodes = ns.NodeContainer()
    nodes.Create(4)
    phone, sat, gs, server = (nodes.Get(i) for i in range(4))

    ns.InternetStackHelper().Install(nodes)

    # ── Service link: Phone <-> Satellite ─────────────────────────────────────
    p2p = ns.PointToPointHelper()
    p2p.SetDeviceAttribute("DataRate",
                           ns.StringValue("10Mbps"))
    p2p.SetChannelAttribute("Delay",
                            ns.StringValue(f"{first['delay_ms']:.3f}ms"))
    devs_svc = p2p.Install(_nc2(phone, sat))

    # RT-calibrated error model on the downlink (satellite → phone)
    em = ns.CreateObject[ns.RateErrorModel]()
    em.SetAttribute("ErrorRate", ns.DoubleValue(first["per"]))
    em.SetAttribute("ErrorUnit", ns.StringValue("ERROR_UNIT_PACKET"))
    devs_svc.Get(1).SetAttribute("ReceiveErrorModel", ns.PointerValue(em))

    # ── Feeder link: Satellite <-> GroundStation (Ka-band ~80° elev) ──────────
    fdr_delay = _one_way_delay_ms(SAT_HEIGHT_M, 80.0)
    p2p.SetDeviceAttribute("DataRate",   ns.StringValue("100Mbps"))
    p2p.SetChannelAttribute("Delay",     ns.StringValue(f"{fdr_delay:.3f}ms"))
    devs_fdr = p2p.Install(_nc2(sat, gs))

    # ── Terrestrial last mile: GroundStation <-> Server ───────────────────────
    p2p.SetDeviceAttribute("DataRate",   ns.StringValue("1Gbps"))
    p2p.SetChannelAttribute("Delay",     ns.StringValue("10ms"))
    devs_gnd = p2p.Install(_nc2(gs, server))

    # ── IP addressing ─────────────────────────────────────────────────────────
    ipv4 = ns.Ipv4AddressHelper()

    ipv4.SetBase(ns.Ipv4Address("10.1.1.0"), ns.Ipv4Mask("255.255.255.0"))
    iface_svc = ipv4.Assign(devs_svc)

    ipv4.SetBase(ns.Ipv4Address("10.1.2.0"), ns.Ipv4Mask("255.255.255.0"))
    ipv4.Assign(devs_fdr)

    ipv4.SetBase(ns.Ipv4Address("10.1.3.0"), ns.Ipv4Mask("255.255.255.0"))
    iface_gnd = ipv4.Assign(devs_gnd)

    ns.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

    # ── Satellite mobility: LEO orbit at ~7.6 km/s ────────────────────────────
    # All non-satellite nodes start stationary
    static_mob = ns.MobilityHelper()
    static_mob.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    static_mob.Install(nodes)

    # Each satellite gets a ConstantVelocity model; here we aggregate
    # it on the single NS-3 'sat' node to move it during the sim.
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

    import collections
    handover_queue = collections.deque(schedule[1:])   # remaining handovers
    handover_count = [0]

    def _handover_tick():
        """
        Single handover callback — applies the next PER from the queue and
        reschedules itself for the subsequent handover time (if any).
        """
        if not handover_queue:
            return
        slot = handover_queue.popleft()
        em.SetAttribute("ErrorRate", ns.DoubleValue(slot["per"]))
        handover_count[0] += 1
        print(f"  [t={ns.Simulator.Now().GetSeconds():.1f}s] "
              f"Handover → sat{slot['sat_id']}  "
              f"PER={slot['per']:.3f}  elev={slot['elev_deg']:.1f}°")
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

    if proto == "udp":
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
        protocol        = proto,
        label           = label,
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

    print(f"\n  Results ({label}):")
    for k, v in result.items():
        print(f"    {k:<24s}: {v}")

    return result


# =============================================================================
# Protocol suite runner
# =============================================================================

def run_ns3_protocol_suite(channel_stats: list,
                            scenario: str = "urban") -> list:
    """
    Run the NS-3 simulation once per entry in config.PROTOCOLS and
    return all results for plotting.

    Parameters
    ----------
    channel_stats : list[dict]
        RT channel statistics from rt_sim.run_ray_tracing().
    scenario : str
        NTN propagation scenario label (used for display only).

    Returns
    -------
    list[dict]  One result dict per protocol (order matches PROTOCOLS).
    """
    results = []
    for pcfg in PROTOCOLS:
        # Augment the protocol config with global TCP settings
        full_cfg = dict(
            tcp_snd_buf   = TCP_SNDRCV_BUF_BYTES,
            tcp_rcv_buf   = TCP_SNDRCV_BUF_BYTES,
            tcp_sack      = TCP_SACK_ENABLED,
            tcp_timestamps = TCP_TIMESTAMPS,
            **pcfg,
        )
        results.append(run_ns3(scenario, full_cfg, channel_stats))

    return results
