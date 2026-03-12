"""
ntn_link_sim.py — DEPRECATED
==============================
This monolithic script has been superseded by the modular layout:

  config.py       — shared parameters
  ntn_phy.py      — Part 1: Sionna 1.2.1 + OpenNTN BER/BLER simulation
  ntn_ns3.py      — Part 2: NS-3 packet-level network simulation
  rt_sim.py       — Part 3: Sionna RT ray tracing (Munich scene)
  main.py         — entry point that runs all three parts

Run ``python main.py`` instead.

----------------------------------------------------------------------
NTN Satellite Link Simulation (original monolithic version)
==============================
Uses the real APIs of:

  Sionna 1.2.1          sionna.phy.*
  OpenNTN (main branch) sionna.phy.channel.tr38811.*
                          ^^^ OpenNTN installs itself INTO Sionna's channel
                              directory via install.sh — it is NOT a
                              standalone package. After install.sh runs, the
                              tr38811 module lives at:
                              <venv>/lib/pythonX.Y/site-packages/sionna/phy/channel/tr38811/
  NS-3                  ns.* Python bindings

Topology
--------
                  feeder link (Ka-band, 26 GHz)
  Internet --- GroundStation ------------------------ Satellite (LEO 600 km)
                                                            |
                                            service link (S-band, 2 GHz)
                                                            |
                                                       Phone (UE)

Installation reminder
---------------------
  # Sionna
  pip install sionna tensorflow

  # OpenNTN  (installs tr38811 INTO sionna/phy/channel/)
  git clone https://github.com/ant-uni-bremen/OpenNTN
  cd OpenNTN && . install.sh      # Linux only; see README for Windows steps

  # NS-3 with Python bindings
  git clone https://gitlab.com/nsnam/ns-3-dev
  cd ns-3-dev
  ./ns3 configure --enable-python-bindings
  ./ns3 build
  export PYTHONPATH=$PYTHONPATH:$(pwd)/build/bindings/python
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# =============================================================================
# Sionna 1.2.1 (sionna.phy.* namespace, changed from 0.x sionna.*)
# =============================================================================
import sionna
import sionna.phy
from sionna.phy import Block
from sionna.phy.ofdm import (
    ResourceGrid,
    ResourceGridMapper,
    LSChannelEstimator,
    LMMSEEqualizer,
    RemoveNulledSubcarriers,
)
from sionna.phy.channel import OFDMChannel
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.mimo import StreamManagement
from sionna.phy.utils import ebnodb2no, compute_ber
from sionna.phy.channel.tr38811 import AntennaArray  # tr38811 version supports satellite patterns

# =============================================================================
# OpenNTN — imported from sionna.phy.channel.tr38811
#
# OpenNTN's install.sh copies the tr38811 package into Sionna's channel
# directory and patches channel/__init__.py with "from . import tr38811".
# After installation the classes live at sionna.phy.channel.tr38811.
#
# API mirrors TR38.901 exactly, with two extra constructor args:
#   bs_height       — satellite altitude in metres
#   elevation_angle — elevation angle from UE to satellite in degrees
#
# gen_single_sector_topology is imported from sionna.phy.channel (not tr38811).
# =============================================================================
from sionna.phy.channel.tr38811 import (
    DenseUrban,
    Urban,
    SubUrban,
)

# =============================================================================
# NS-3 Python bindings
# =============================================================================
# NS-3 cppyy-based bindings (3.37+): correct import is `from ns import ns`
# Everything lives flat on the ns object: ns.NodeContainer, ns.Simulator, etc.
# If you see [runStaticInitializersOnce] warnings, pin: pip install cppyy==2.4.2
from ns import ns

def _nc2(a, b):
    """NodeContainer from two nodes — cppyy doesn't support the 2-arg constructor."""
    nc = ns.NodeContainer()
    nc.Add(a)
    nc.Add(b)
    return nc



# =============================================================================
# Parameters
# =============================================================================

CARRIER_FREQ_HZ     = 2.0e9        # S-band service link
SUBCARRIER_SPACING  = 15e3         # NR numerology 0 (Hz)
FFT_SIZE            = 128
NUM_OFDM_SYMBOLS    = 14           # 1 NR slot
CP_LENGTH           = 9
PILOT_SYMBOL_IDX    = [2, 11]
NUM_BITS_PER_SYMBOL = 2            # QPSK
CODERATE            = 0.5
BATCH_SIZE          = 64

SAT_HEIGHT_M        = 600_000.0    # 600 km LEO in metres; divided by 1e3 when passed to OpenNTN
ELEVATION_ANGLE_DEG = 60.0

SIM_DURATION_S      = 10.0
APP_DATA_RATE       = "2Mbps"
PACKET_SIZE_BYTES   = 1400


# =============================================================================
# Part 1 – Sionna 1.2.1 + OpenNTN: OFDM BER/BLER link simulation
# =============================================================================

def build_channel_model(scenario: str):
    """
    Construct an OpenNTN TR38.811 channel model.

    Constructor signature (mirrors Sionna's TR38.901 UMa/UMi/RMa):
        ChannelClass(
            carrier_frequency,
            ut_array,
            bs_array,
            direction,          "uplink" | "downlink"
            elevation_angle,    UE->satellite elevation angle [deg]  <-- NTN-specific
            enable_pathloss,
            enable_shadow_fading,
        )
    """
    # Satellite antenna — circular aperture pattern (3GPP TR 38.811 §6.4)
    sat_array = AntennaArray(
        num_rows=1,
        num_cols=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=CARRIER_FREQ_HZ,
    )

    # UE (phone) antenna — quasi-isotropic
    ue_array = AntennaArray(
        num_rows=1,
        num_cols=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=CARRIER_FREQ_HZ,
    )

    cls = {"dense_urban": DenseUrban, "urban": Urban, "suburban": SubUrban}[scenario]

    return cls(
        carrier_frequency=CARRIER_FREQ_HZ,
        ut_array=ue_array,
        bs_array=sat_array,
        direction="downlink",
        elevation_angle=ELEVATION_ANGLE_DEG,
        enable_pathloss=True,
        enable_shadow_fading=True,
    )


class NTNOFDMModel(Block):
    """
    Sionna 1.2.1 end-to-end NTN downlink model (SISO QPSK LDPC).

    Chain:
      BinarySource → LDPC5GEncoder → Mapper → ResourceGridMapper
        → OFDMChannel (OpenNTN TR38.811 channel)
        → LSChannelEstimator → LMMSEEqualizer
        → Demapper → LDPC5GDecoder
    """

    def __init__(self, channel_model):
        super().__init__()

        # 1 BS (satellite beam), 1 UT (phone), 1 stream
        self._sm = StreamManagement(
            rx_tx_association=np.array([[1]]),
            num_streams_per_tx=1,
        )

        self._rg = ResourceGrid(
            num_ofdm_symbols=NUM_OFDM_SYMBOLS,
            fft_size=FFT_SIZE,
            subcarrier_spacing=SUBCARRIER_SPACING,
            num_tx=1,
            num_streams_per_tx=1,
            cyclic_prefix_length=CP_LENGTH,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=PILOT_SYMBOL_IDX,
        )

        self._n = int(self._rg.num_data_symbols * NUM_BITS_PER_SYMBOL)
        self._k = int(self._n * CODERATE)

        self._src     = BinarySource()
        self._enc     = LDPC5GEncoder(self._k, self._n)
        self._mapper  = Mapper("qam", NUM_BITS_PER_SYMBOL)
        self._rg_map  = ResourceGridMapper(self._rg)

        # OFDMChannel accepts any Sionna-compatible channel model,
        # including OpenNTN's Urban/DenseUrban/SubUrban
        self._channel = OFDMChannel(
            channel_model=channel_model,
            resource_grid=self._rg,
            add_awgn=True,
            normalize_channel=True,
            return_channel=True,
        )

        self._rm_null = RemoveNulledSubcarriers(self._rg)
        self._ls_est  = LSChannelEstimator(self._rg, interpolation_type="nn")
        self._lmmse   = LMMSEEqualizer(self._rg, self._sm)
        self._demap   = Demapper("app", "qam", NUM_BITS_PER_SYMBOL)
        self._dec     = LDPC5GDecoder(self._enc, hard_out=True)

    @tf.function(jit_compile=False)
    def call(self, batch_size, ebno_db):
        no   = ebnodb2no(ebno_db, NUM_BITS_PER_SYMBOL, CODERATE, self._rg)

        b    = self._src([batch_size, 1, 1, self._k])
        cw   = self._enc(b)
        syms = self._mapper(cw)
        x    = self._rg_map(syms)

        y, _ = self._channel(x, no)                # Sionna 1.2: separate args, not list

        h_hat, err_var = self._ls_est(y, no)
        x_hat, no_eff  = self._lmmse(y, h_hat, err_var, no)
        llr            = self._demap(x_hat, no_eff)
        b_hat          = self._dec(llr)

        return b, b_hat


def run_sionna_ber(snr_db_range: np.ndarray, scenario: str):
    print(f"\n[Sionna 1.2.1 + OpenNTN]  {scenario.upper()}"
          f"  LEO {SAT_HEIGHT_M/1e3:.0f} km  elev {ELEVATION_ANGLE_DEG:.0f}deg\n")

    sionna.phy.config.seed = 42

    ch_model = build_channel_model(scenario)

    # For NTN, elevation_angle is fixed in the constructor.
    # set_topology needs geometrically consistent position tensors.
    # Satellite directly above the UE at LEO altitude.
    ut_loc          = tf.zeros([BATCH_SIZE, 1, 3], dtype=tf.float32)
    bs_loc          = tf.tile(tf.constant([[[0., 0., SAT_HEIGHT_M]]], dtype=tf.float32), [BATCH_SIZE, 1, 1])
    ut_orientations = tf.zeros([BATCH_SIZE, 1, 3], dtype=tf.float32)
    bs_orientations = tf.zeros([BATCH_SIZE, 1, 3], dtype=tf.float32)
    ut_velocities   = tf.tile(tf.constant([[[7600., 0., 0.]]], dtype=tf.float32), [BATCH_SIZE, 1, 1])
    in_state        = tf.zeros([BATCH_SIZE, 1], dtype=tf.bool)

    # Pass positionally — internal scenario wrapper re-passes args positionally,
    # so order must match system_level_scenario.set_topology exactly.
    # in_state cast to bool to survive internal tf.where.
    ch_model.set_topology(
        ut_loc,
        bs_loc,
        ut_orientations,
        bs_orientations,
        ut_velocities,
        tf.cast(in_state, tf.bool),
    )

    model = NTNOFDMModel(ch_model)

    ber_arr, bler_arr = [], []
    for ebno_db in snr_db_range:
        b, b_hat = model(BATCH_SIZE, tf.cast(ebno_db, tf.float32))
        ber  = float(compute_ber(b, b_hat).numpy())
        bler = float(tf.reduce_mean(
            tf.cast(tf.reduce_any(tf.not_equal(b, b_hat), axis=-1),
                    tf.float32)).numpy())
        ber_arr.append(ber)
        bler_arr.append(bler)
        print(f"  Eb/N0 = {ebno_db:5.1f} dB   BER = {ber:.5f}   BLER = {bler:.5f}")

    return np.array(ber_arr), np.array(bler_arr)


# =============================================================================
# Part 2 – NS-3: Packet-level NTN network simulation
# =============================================================================

# Propagation delay for a LEO at given height and elevation angle
# (simple geometry — same formula OpenNTN uses internally)
def _slant_range_m(height_m: float, elev_deg: float) -> float:
    RE = 6_371_000.0
    e  = np.radians(elev_deg)
    return (np.sqrt((RE + height_m)**2 - (RE * np.cos(e))**2)
            - RE * np.sin(e))

def _delay_ms(height_m: float, elev_deg: float) -> float:
    return _slant_range_m(height_m, elev_deg) / 3e8 * 1e3

def _loss_pct(height_m: float, elev_deg: float,
              fc_hz: float = CARRIER_FREQ_HZ) -> float:
    """
    Approximate packet loss from free-space path loss + simplified link budget.
    Tx EIRP 30 dBm, quasi-iso Rx, thermal noise floor -120 dBm.
    """
    d   = _slant_range_m(height_m, elev_deg)
    pl  = 20 * np.log10(4 * np.pi * d * fc_hz / 3e8)
    snr = 30.0 - pl + 120.0
    per = 1.0 / (1.0 + np.exp((snr - 7.5) * 0.7))
    return float(np.clip(per * 100.0, 0.0, 99.0))


def run_ns3(scenario: str, elev_deg: float):
    print(f"\n[NS-3]  {scenario.upper()}  elev={elev_deg:.0f}deg  "
          f"height={SAT_HEIGHT_M/1e3:.0f} km\n")

    # ── 4 nodes: Phone | Satellite | GroundStation | Server ──────────────────
    nodes = ns.NodeContainer()
    nodes.Create(4)
    phone, sat, gs, server = (nodes.Get(i) for i in range(4))

    ns.InternetStackHelper().Install(nodes)

    # ── Service link: Phone <-> Satellite (S-band, NTN delay + PER) ──────────
    svc_delay = _delay_ms(SAT_HEIGHT_M, elev_deg)
    svc_loss  = _loss_pct(SAT_HEIGHT_M, elev_deg)

    p2p = ns.PointToPointHelper()
    p2p.SetDeviceAttribute("DataRate",   ns.StringValue("10Mbps"))
    p2p.SetChannelAttribute("Delay",     ns.StringValue(f"{svc_delay:.3f}ms"))
    devs_svc = p2p.Install(_nc2(phone, sat))

    em = ns.CreateObject[ns.RateErrorModel]()
    em.SetAttribute("ErrorRate", ns.DoubleValue(svc_loss / 100.0))
    em.SetAttribute("ErrorUnit", ns.StringValue("ERROR_UNIT_PACKET"))
    devs_svc.Get(1).SetAttribute("ReceiveErrorModel", ns.PointerValue(em))

    # ── Feeder link: Satellite <-> GroundStation (Ka-band, high elev) ─────────
    fdr_delay = _delay_ms(SAT_HEIGHT_M, 80.0)
    p2p.SetDeviceAttribute("DataRate",   ns.StringValue("100Mbps"))
    p2p.SetChannelAttribute("Delay",     ns.StringValue(f"{fdr_delay:.3f}ms"))
    p2p.Install(_nc2(sat, gs))

    # ── Terrestrial last mile: GroundStation <-> Server (~10 ms fibre) ────────
    p2p.SetDeviceAttribute("DataRate",   ns.StringValue("1Gbps"))
    p2p.SetChannelAttribute("Delay",     ns.StringValue("10ms"))
    devs_gnd = p2p.Install(_nc2(gs, server))

    # ── IP addressing ─────────────────────────────────────────────────────────
    ipv4 = ns.Ipv4AddressHelper()

    ipv4.SetBase(ns.Ipv4Address("10.1.1.0"),
                 ns.Ipv4Mask("255.255.255.0"))
    ipv4.Assign(devs_svc)

    ipv4.SetBase(ns.Ipv4Address("10.1.2.0"),
                 ns.Ipv4Mask("255.255.255.0"))
    p2p.SetDeviceAttribute("DataRate",   ns.StringValue("100Mbps"))
    p2p.SetChannelAttribute("Delay",     ns.StringValue(f"{fdr_delay:.3f}ms"))
    sat_gs_devs = p2p.Install(_nc2(sat, gs))
    ipv4.Assign(sat_gs_devs)

    ipv4.SetBase(ns.Ipv4Address("10.1.3.0"),
                 ns.Ipv4Mask("255.255.255.0"))
    iface_gnd = ipv4.Assign(devs_gnd)

    ns.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

    # ── Satellite mobility: LEO orbit at ~7.6 km/s ────────────────────────────
    # All non-satellite nodes get ConstantPosition (stationary)
    static_mob = ns.MobilityHelper()
    static_mob.SetMobilityModel('ns3::ConstantPositionMobilityModel')
    static_mob.Install(nodes)

    # Satellite gets ConstantVelocity — must aggregate it explicitly
    sat_mob = ns.CreateObject[ns.ConstantVelocityMobilityModel]()
    sat.AggregateObject(sat_mob)
    sat_mob.SetPosition(ns.Vector(0.0, 0.0, SAT_HEIGHT_M))
    sat_mob.SetVelocity(ns.Vector(7600.0, 0.0, 0.0))

    # ── UDP OnOff app: Phone → Server ─────────────────────────────────────────
    port = 9
    sink_addr = ns.InetSocketAddress(iface_gnd.GetAddress(1), port)

    onoff = ns.OnOffHelper(
        "ns3::UdpSocketFactory", sink_addr.ConvertTo())
    onoff.SetAttribute("DataRate",   ns.StringValue(APP_DATA_RATE))
    onoff.SetAttribute("PacketSize", ns.UintegerValue(PACKET_SIZE_BYTES))
    onoff.SetAttribute("OnTime",
                       ns.StringValue("ns3::ConstantRandomVariable[Constant=1]"))
    onoff.SetAttribute("OffTime",
                       ns.StringValue("ns3::ConstantRandomVariable[Constant=0]"))
    tx = onoff.Install(phone)
    tx.Start(ns.Seconds(1.0))
    tx.Stop(ns.Seconds(SIM_DURATION_S))

    sink = ns.PacketSinkHelper(
        "ns3::UdpSocketFactory",
        ns.InetSocketAddress(ns.Ipv4Address.GetAny(), port).ConvertTo())
    rx = sink.Install(server)
    rx.Start(ns.Seconds(0.0))
    rx.Stop(ns.Seconds(SIM_DURATION_S + 1.0))

    # ── FlowMonitor ───────────────────────────────────────────────────────────
    fm_helper = ns.FlowMonitorHelper()
    monitor   = fm_helper.InstallAll()

    # ── Run ───────────────────────────────────────────────────────────────────
    ns.Simulator.Stop(ns.Seconds(SIM_DURATION_S + 2.0))
    ns.Simulator.Run()
    monitor.CheckForLostPackets()

    stats      = monitor.GetFlowStats()
    classifier = fm_helper.GetClassifier()

    result = dict(scenario=scenario, elevation_deg=elev_deg,
                  svc_delay_ms=round(svc_delay, 2),
                  svc_loss_pct=round(svc_loss, 2),
                  tx_packets=0, rx_packets=0, loss_pct=0.0,
                  mean_delay_ms=0.0, jitter_ms=0.0, throughput_kbps=0.0)

    for pair in stats:
        fid, fs = pair.first, pair.second
        if classifier.FindFlow(fid).protocol != 17:   # UDP only
            continue
        if fs.rxPackets == 0:
            continue
        rx_n = int(fs.rxPackets)
        tx_n = int(fs.txPackets)
        result.update(
            tx_packets      = tx_n,
            rx_packets      = rx_n,
            loss_pct        = round(100.0 * (1 - rx_n / max(tx_n, 1)), 2),
            mean_delay_ms   = round(fs.delaySum.GetSeconds()  / rx_n  * 1e3, 2),
            jitter_ms       = round(fs.jitterSum.GetSeconds() / max(rx_n - 1, 1) * 1e3, 2),
            throughput_kbps = round(fs.rxBytes * 8.0 / (SIM_DURATION_S - 1.0) / 1e3, 2),
        )

    ns.Simulator.Destroy()

    for k, v in result.items():
        print(f"  {k:<24s}: {v}")
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  NTN Satellite Link Simulation")
    print("  Sionna 1.2.1 + OpenNTN (TR38.811) + NS-3")
    print("=" * 70)

    snr_range   = np.arange(0, 22, 2, dtype=float)
    ber_results = {}
    for sc in ["urban", "dense_urban", "suburban"]:
        ber, bler       = run_sionna_ber(snr_range, sc)
        ber_results[sc] = (ber, bler)

    ns3_results = []
    for elev, sc in [(80, "urban"), (45, "dense_urban"), (20, "suburban")]:
        ns3_results.append(run_ns3(sc, float(elev)))

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "NTN Satellite Simulation  (Sionna 1.2.1 + OpenNTN TR38.811 + NS-3)\n"
        f"LEO {SAT_HEIGHT_M/1e3:.0f} km  |  S-band {CARRIER_FREQ_HZ/1e9:.1f} GHz  |"
        f"  Elevation {ELEVATION_ANGLE_DEG:.0f}deg  |  QPSK  r={CODERATE}  LDPC",
        fontsize=10,
    )
    colors = {"urban": "#1f77b4", "dense_urban": "#d62728", "suburban": "#2ca02c"}

    ax = axes[0]
    for sc, (ber, _) in ber_results.items():
        ax.semilogy(snr_range, np.clip(ber, 1e-5, 1), "o-", ms=5,
                    label=sc.replace("_", " "), color=colors[sc])
    ax.set_xlabel("Eb/N0 (dB)"); ax.set_ylabel("Coded BER")
    ax.set_title("[Sionna + OpenNTN]  Coded BER\n(QPSK LDPC r=0.5, LS+LMMSE)")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.4)
    ax.set_ylim([1e-5, 1])

    ax = axes[1]
    for sc, (_, bler) in ber_results.items():
        ax.semilogy(snr_range, np.clip(bler, 1e-5, 1), "s-", ms=5,
                    label=sc.replace("_", " "), color=colors[sc])
    ax.set_xlabel("Eb/N0 (dB)"); ax.set_ylabel("BLER")
    ax.set_title("[Sionna + OpenNTN]  BLER\n(QPSK LDPC r=0.5, LS+LMMSE)")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.4)
    ax.set_ylim([1e-5, 1])

    ax  = axes[2]
    ax2 = ax.twinx()
    labels = [f"{r['elevation_deg']:.0f}deg\n{r['scenario'].replace('_',' ')}"
              for r in ns3_results]
    x = np.arange(len(labels))
    ax.bar(x - 0.2,  [r["mean_delay_ms"]    for r in ns3_results], 0.35,
           color=["#1f77b4", "#d62728", "#2ca02c"], alpha=0.8,
           label="Latency (ms)")
    ax2.bar(x + 0.2, [r["throughput_kbps"]  for r in ns3_results], 0.35,
            color=["#aec7e8", "#ffbb78", "#98df8a"], alpha=0.8,
            label="Throughput (kbps)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Latency (ms)"); ax2.set_ylabel("Throughput (kbps)")
    ax.set_title("[NS-3]  Latency & Throughput\nPhone -> Sat -> GS -> Internet")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.4)

    plt.tight_layout()
    out = "ntn_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n[Plot] Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()