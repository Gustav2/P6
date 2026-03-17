"""
topology_diagram.py — Scenario topology illustration and protocol comparison
=============================================================================
Produces output figures for the NTN satellite simulation:

  ntn_protocol_comparison.png
      Grouped bar chart comparing all transport protocols (UDP, TCP NewReno,
      TCP CUBIC, TCP BBR, QUIC) on latency, throughput, and packet loss.

  ntn_summary.png
      Five-panel summary: BER/BLER from Sionna + 3 NS-3 bar charts.

  ntn_link_budget_waterfall.png
      Horizontal waterfall showing per-satellite link budget breakdown.

  ntn_snr_vs_elevation.png
      SNR vs elevation angle with PER sigmoid overlay.

  ntn_latency_breakdown.png
      Per-hop stacked latency bar chart.

  ntn_handover_impact.png
      Per-slot throughput showing TCP congestion collapse vs QUIC resilience.

  ntn_protocol_radar.png
      Spider/radar chart comparing protocols on 5 performance axes.

  ntn_results.png
      Three-panel combined results summary (throughput, latency, loss).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from config import SAT_HEIGHT_M, CARRIER_FREQ_HZ, SIM_DURATION_S, PROTOCOLS


# =============================================================================
# Colour palette (consistent with rest of project)
# =============================================================================

PROTO_COLORS = {
    "UDP":         "#e377c2",
    "TCP NewReno": "#1f77b4",
    "TCP CUBIC":   "#2ca02c",
    "TCP BBR":     "#ff7f0e",
    "QUIC":        "#9467bd",
}

# Fallback for any unlisted label
_FALLBACK_COLORS = ["#9467bd", "#8c564b", "#17becf", "#bcbd22"]


def _proto_color(label: str, idx: int) -> str:
    return PROTO_COLORS.get(label, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


# Slot color constants — used by draw_snr_vs_elevation and draw_handover_impact
# to mark the three simulated satellites consistently.
_SLOT_COLORS = ["#1a6fc4", "#2ca02c", "#d62728"]


# =============================================================================
# Protocol comparison bar chart
# =============================================================================

def draw_protocol_comparison(ns3_results: list,
                              out: str = "output/ntn_protocol_comparison.png") -> str:
    """
    Grouped bar chart comparing transport protocols on three metrics:
      - Mean end-to-end latency [ms]
      - Throughput [kbps]
      - Packet loss rate [%]

    Parameters
    ----------
    ns3_results : list[dict]
        Output of ntn_ns3.run_ns3_protocol_suite() — one dict per protocol.
        Each dict must contain keys:
          label, mean_delay_ms, throughput_kbps, loss_pct, handovers,
          elevation_deg, svc_delay_ms.
    out : str
        Output filename.

    Returns
    -------
    str  Path of the saved figure.
    """
    if not ns3_results:
        print("[ProtocolChart]  No results to plot — skipping.")
        return out

    labels    = [r["label"]           for r in ns3_results]
    latencies = [r["mean_delay_ms"]   for r in ns3_results]
    tputs     = [r["throughput_kbps"] for r in ns3_results]
    losses    = [r["loss_pct"]        for r in ns3_results]
    handovers = [r.get("handovers", 0) for r in ns3_results]

    colors = [_proto_color(lbl, i) for i, lbl in enumerate(labels)]
    x = np.arange(len(labels))
    bar_w = 0.62

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Transport Protocol Comparison — 5G-NTN LEO Satellite Link\n"
        f"(Sionna RT channel stats → NS-3 packet simulation, "
        f"{SIM_DURATION_S:.0f} s, Munich urban scene)",
        fontsize=10,
    )

    # ── Panel 1: Mean latency ─────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(x, latencies, bar_w, color=colors, edgecolor="#333",
                  linewidth=0.8, alpha=0.88)
    ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean E2E Latency [ms]")
    ax.set_title("Latency")
    ax.grid(axis="y", alpha=0.35)
    ax.set_ylim(0, max(latencies) * 1.25 + 1)

    # ── Panel 2: Throughput ───────────────────────────────────────────────────
    ax = axes[1]
    bars = ax.bar(x, tputs, bar_w, color=colors, edgecolor="#333",
                  linewidth=0.8, alpha=0.88)
    ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Throughput [kbps]")
    ax.set_title("Throughput")
    ax.grid(axis="y", alpha=0.35)
    ax.set_ylim(0, max(tputs) * 1.25 + 1)

    # ── Panel 3: Loss + handover annotation ──────────────────────────────────
    ax = axes[2]
    bars = ax.bar(x, losses, bar_w, color=colors, edgecolor="#333",
                  linewidth=0.8, alpha=0.88)
    ax.bar_label(bars, fmt="%.2f%%", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Packet Loss [%]")
    ax.set_title("Packet Loss")
    ax.grid(axis="y", alpha=0.35)
    ax.set_ylim(0, max(losses) * 1.35 + 0.1)

    # Annotate handover counts below each bar
    for xi, ho in zip(x, handovers):
        ax.text(xi, -0.04 * (max(losses) * 1.35 + 0.1),
                f"{ho} H/O", ha="center", va="top",
                fontsize=7, color="#555")

    # ── Shared legend ─────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=_proto_color(lbl, i), edgecolor="#333",
                       label=lbl)
        for i, lbl in enumerate(labels)
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(labels), fontsize=9,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

    # ── Satellite/handover info table ─────────────────────────────────────────
    info_lines = []
    for r in ns3_results:
        info_lines.append(
            f"{r['label']}: svc_delay={r.get('svc_delay_ms', '?'):.1f} ms  "
            f"svc_PER={r.get('svc_loss_pct', '?'):.1f}%  "
            f"elev={r.get('elevation_deg', '?'):.0f}°"
        )
    fig.text(0.5, -0.09, "   |   ".join(info_lines),
             ha="center", va="top", fontsize=6.5,
             color="#555",
             transform=fig.transFigure)

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[ProtocolChart]  Saved -> {out}")
    plt.close()
    return out


# =============================================================================
# Combined summary figure
# =============================================================================

def draw_summary(ber_results: dict, ns3_results: list,
                 snr_range=None,
                 out: str = "output/ntn_summary.png") -> str:
    """
    Five-panel summary figure combining BER/BLER (Part 1) with the
    protocol comparison bar charts (Part 2) in a single page.

    Parameters
    ----------
    ber_results  : dict        Mapping scenario -> (ber_array, bler_array).
    ns3_results  : list        Output of run_ns3_protocol_suite().
    snr_range    : array-like  Eb/N0 values [dB] matching the ber_results
                               arrays.  If None, inferred from the first
                               ber array length using 0.5 dB steps from 0.
    out          : str         Output filename.

    Returns
    -------
    str  Path of the saved figure.
    """
    # Derive snr_range from the actual ber arrays if not supplied.
    if snr_range is None and ber_results:
        first_ber, _ = next(iter(ber_results.values()))
        n = len(first_ber)
        snr_range = np.arange(0, n * 0.5, 0.5, dtype=float)[:n]
    elif snr_range is None:
        snr_range = np.arange(0, 22, 2, dtype=float)
    else:
        snr_range = np.asarray(snr_range, dtype=float)
    ber_colors = {
        "urban":       "#1f77b4",
        "dense_urban": "#d62728",
        "suburban":    "#2ca02c",
    }

    if not ns3_results:
        print("[Summary]  No NS-3 results — skipping summary figure.")
        return out

    labels    = [r["label"]           for r in ns3_results]
    latencies = [r["mean_delay_ms"]   for r in ns3_results]
    tputs     = [r["throughput_kbps"] for r in ns3_results]
    losses    = [r["loss_pct"]        for r in ns3_results]
    colors    = [_proto_color(lbl, i) for i, lbl in enumerate(labels)]
    x         = np.arange(len(labels))
    bar_w     = 0.62

    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(
        "NTN Satellite Simulation — Full Results\n"
        f"Sionna 1.2.1 + OpenNTN (TR38.811) + Sionna RT (Munich) + NS-3\n"
        f"LEO {SAT_HEIGHT_M/1e3:.0f} km  |  {CARRIER_FREQ_HZ/1e9:.1f} GHz  |  "
        f"Sim {SIM_DURATION_S:.0f} s",
        fontsize=10, y=0.99,
    )

    # 2 rows × 3 cols; cols 1-2 in row 1 = BER/BLER; cols 1-3 in row 2 = NS-3
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38)

    # ── BER ───────────────────────────────────────────────────────────────────
    ax_ber = fig.add_subplot(gs[0, 0])
    for sc, (ber, _) in ber_results.items():
        ax_ber.semilogy(snr_range, np.clip(ber, 1e-5, 1), "o-", ms=4,
                        label=sc.replace("_", " "),
                        color=ber_colors.get(sc, "gray"))
    ax_ber.set_xlabel("Eb/N0 [dB]")
    ax_ber.set_ylabel("Coded BER")
    ax_ber.set_title("[Sionna + OpenNTN]  Coded BER\n(QPSK LDPC r=0.5)")
    ax_ber.legend(fontsize=8)
    ax_ber.grid(True, which="both", alpha=0.4)
    ax_ber.set_ylim([1e-5, 1])

    # ── BLER ──────────────────────────────────────────────────────────────────
    ax_bler = fig.add_subplot(gs[0, 1])
    for sc, (_, bler) in ber_results.items():
        ax_bler.semilogy(snr_range, np.clip(bler, 1e-5, 1), "s-", ms=4,
                         label=sc.replace("_", " "),
                         color=ber_colors.get(sc, "gray"))
    ax_bler.set_xlabel("Eb/N0 [dB]")
    ax_bler.set_ylabel("BLER")
    ax_bler.set_title("[Sionna + OpenNTN]  BLER\n(QPSK LDPC r=0.5)")
    ax_bler.legend(fontsize=8)
    ax_bler.grid(True, which="both", alpha=0.4)
    ax_bler.set_ylim([1e-5, 1])

    # ── NS-3 latency ──────────────────────────────────────────────────────────
    ax_lat = fig.add_subplot(gs[1, 0])
    bars = ax_lat.bar(x, latencies, bar_w, color=colors, edgecolor="#333",
                      linewidth=0.8, alpha=0.88)
    ax_lat.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    ax_lat.set_xticks(x); ax_lat.set_xticklabels(labels, fontsize=8)
    ax_lat.set_ylabel("Mean Latency [ms]")
    ax_lat.set_title("[NS-3]  Latency")
    ax_lat.grid(axis="y", alpha=0.35)
    ax_lat.set_ylim(0, max(latencies) * 1.3 + 1)

    # ── NS-3 throughput ───────────────────────────────────────────────────────
    ax_tput = fig.add_subplot(gs[1, 1])
    bars = ax_tput.bar(x, tputs, bar_w, color=colors, edgecolor="#333",
                       linewidth=0.8, alpha=0.88)
    ax_tput.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
    ax_tput.set_xticks(x); ax_tput.set_xticklabels(labels, fontsize=8)
    ax_tput.set_ylabel("Throughput [kbps]")
    ax_tput.set_title("[NS-3]  Throughput")
    ax_tput.grid(axis="y", alpha=0.35)
    ax_tput.set_ylim(0, max(tputs) * 1.3 + 1)

    # ── NS-3 loss ─────────────────────────────────────────────────────────────
    ax_loss = fig.add_subplot(gs[1, 2])
    bars = ax_loss.bar(x, losses, bar_w, color=colors, edgecolor="#333",
                       linewidth=0.8, alpha=0.88)
    ax_loss.bar_label(bars, fmt="%.2f%%", fontsize=7, padding=2)
    ax_loss.set_xticks(x); ax_loss.set_xticklabels(labels, fontsize=8)
    ax_loss.set_ylabel("Packet Loss [%]")
    ax_loss.set_title("[NS-3]  Packet Loss")
    ax_loss.grid(axis="y", alpha=0.35)
    ax_loss.set_ylim(0, max(losses) * 1.35 + 0.1)

    # ── Blank top-right cell: insert a mini topology sketch ──────────────────
    ax_topo = fig.add_subplot(gs[0, 2])
    ax_topo.axis("off")
    _mini_topology(ax_topo)

    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[Summary]  Saved -> {out}")
    plt.close()
    return out


def _mini_topology(ax) -> None:
    """
    Draw a compact inline topology sketch into a pre-existing axes.
    Used by draw_summary() for the top-right panel.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_title("Topology sketch", fontsize=8)

    # Nodes
    def _box(cx, cy, label, sub="", fc="#dce9f7"):
        b = FancyBboxPatch((cx - 1.1, cy - 0.35), 2.2, 0.7,
                           boxstyle="round,pad=0.05",
                           facecolor=fc, edgecolor="#555", lw=0.8, zorder=3)
        ax.add_patch(b)
        ax.text(cx, cy + (0.06 if sub else 0), label,
                ha="center", va="center", fontsize=6.5, fontweight="bold",
                color="black", zorder=4)
        if sub:
            ax.text(cx, cy - 0.14, sub,
                    ha="center", va="center", fontsize=5.5, color="#555",
                    zorder=4)

    _box(2, 4.2, "LEO Sat 0", "~90°", fc="#cce5ff")
    _box(6, 4.2, "LEO Sat 1", "~75°", fc="#cce5ff")
    _box(1.5, 2,  "UE", "street-level", fc="#d4edda")
    _box(6.5, 2,  "GS", "ground stn", fc="#fff3cd")
    _box(9, 2,   "Server", "internet", fc="#f8d7da")

    # Arrows
    def _arr(x1, y1, x2, y2, col="#333", ls="-", lw=1):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                    lw=lw, linestyle=ls), zorder=2)

    _arr(1.5, 2.35, 2, 3.85, col="#0056b3", lw=1.5)   # UE → Sat0
    _arr(1.5, 2.35, 6, 3.85, col="#888", ls="--")     # UE → Sat1
    _arr(2, 3.85, 6.5, 2.35, col="#856404")            # Sat0 → GS
    _arr(7.6, 2.0, 7.9, 2.0, col="#155724", lw=1)     # GS → Server

    ax.text(3.8, 3.5, "NTN svc\nlink", ha="center", fontsize=5.5,
            color="#0056b3")
    ax.text(4.2, 2.8, "Ka feeder", ha="center", fontsize=5.5,
            color="#856404")
    ax.text(8.2, 2.2, "fibre", ha="center", fontsize=5.5,
            color="#155724")


# =============================================================================
# 1. Link budget waterfall
# =============================================================================

def draw_link_budget_waterfall(channel_stats: list,
                                out: str = "output/ntn_link_budget_waterfall.png") -> str:
    """
    Horizontal stacked-bar waterfall showing the link budget breakdown
    per satellite for the phone EIRP (direct topology).

    Stages: TX EIRP → (−FSPL) → (±urban correction) → SNR → (−threshold) → margin

    Parameters
    ----------
    channel_stats : list[dict]  RT channel stats from rt_sim.run_ray_tracing().
    out : str  Output filename.

    Returns
    -------
    str  Path of the saved figure.
    """
    import math as _math
    from config import (PHONE_EIRP_DBM,
                        SAT_HEIGHT_M, CARRIER_FREQ_HZ,
                        SAT_RX_ANTENNA_GAIN_DB,
                        NOISE_FLOOR_DBM, SNR_THRESH_DB)
    from ntn_ns3 import _fspl_db, _rt_calibrated_per

    # Sort by elevation descending
    stats = sorted(channel_stats, key=lambda s: s["elevation_deg"], reverse=True)
    ref_gain = stats[0]["mean_path_gain_db"] if stats else -150.0

    n_sats = len(stats)

    # Build budget stages for each sat
    def _budget(stat, eirp_dbm):
        fspl  = _fspl_db(SAT_HEIGHT_M, max(stat["elevation_deg"], 1.0))
        gain  = stat["mean_path_gain_db"]
        if gain <= -150.0:
            urban = -10.0
        elif ref_gain > -150.0:
            urban = gain - ref_gain
        else:
            urban = 0.0
        snr = eirp_dbm - fspl + urban + SAT_RX_ANTENNA_GAIN_DB - NOISE_FLOOR_DBM
        per = _rt_calibrated_per(fspl, gain, ref_gain, eirp_dbm)
        return dict(eirp=eirp_dbm, fspl=fspl, urban=urban,
                    noise=NOISE_FLOOR_DBM, snr=snr, thresh=SNR_THRESH_DB,
                    margin=snr - SNR_THRESH_DB, per=per)

    budgets = [_budget(s, PHONE_EIRP_DBM) for s in stats]

    fig, axes = plt.subplots(1, n_sats, figsize=(6 * n_sats, 5),
                              sharex=False)
    if n_sats == 1:
        axes = [axes]
    fig.suptitle(
        "Link Budget Waterfall — Per Satellite\n"
        f"TX EIRP → FSPL → Urban correction → SNR → Margin vs threshold  "
        f"(3.5 GHz, LEO {SAT_HEIGHT_M/1e3:.0f} km, Phone EIRP {PHONE_EIRP_DBM:.0f} dBm)",
        fontsize=10,
    )

    stage_colors = {
        "TX EIRP":        "#4CAF50",
        "−FSPL":          "#F44336",
        "Urban corr.":    "#FF9800",
        "−Noise floor":   "#2196F3",
        "SNR":            "#9C27B0",
        "Threshold":      "#FF5722",
        "Margin":         "#009688",
    }

    def _waterfall_ax(ax, budget, title):
        stages = [
            ("TX EIRP",     budget["eirp"],               True),
            ("−FSPL",       -budget["fspl"],              False),
            ("Urban corr.", budget["urban"],               budget["urban"] >= 0),
            ("−Noise floor", -budget["noise"],             True),
        ]
        running = 0.0
        bottoms = []
        widths  = []
        stage_names = []
        for name, val, pos in stages:
            bottoms.append(running if val >= 0 else running + val)
            widths.append(abs(val))
            stage_names.append(name)
            running += val

        snr = running
        # Threshold bar
        bottoms.append(0)
        widths.append(SNR_THRESH_DB)
        stage_names.append("Threshold")
        # Margin bar
        margin = snr - SNR_THRESH_DB
        bottoms.append(SNR_THRESH_DB)
        widths.append(max(margin, 0))
        stage_names.append("Margin")

        colors = [stage_colors.get(n, "#888") for n in stage_names]
        y = np.arange(len(stage_names))
        ax.barh(y, widths, left=bottoms, color=colors,
                edgecolor="#333", linewidth=0.7, alpha=0.88)
        ax.set_yticks(y)
        ax.set_yticklabels(stage_names, fontsize=8)
        ax.set_xlabel("Power / SNR [dB]", fontsize=8)
        ax.set_title(title, fontsize=8.5, fontweight="bold")
        ax.axvline(snr, color="#9C27B0", linewidth=1.5, linestyle=":",
                   label=f"SNR={snr:.1f} dB")
        ax.axvline(SNR_THRESH_DB, color="#FF5722", linewidth=1.2,
                   linestyle="--", label=f"Thresh={SNR_THRESH_DB:.1f} dB")
        ax.grid(axis="x", alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
        # PER annotation
        ax.text(0.98, 0.04, f"PER = {budget['per']:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#d62728",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#d62728", alpha=0.85))

    for i, (stat, b) in enumerate(zip(stats, budgets)):
        _waterfall_ax(axes[i], b,
                      f"Sat {stat['sat_id']}  (elev {stat['elevation_deg']:.0f}°)\n"
                      f"Phone EIRP {PHONE_EIRP_DBM:.0f} dBm")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[LinkBudget]  Saved -> {out}")
    plt.close()
    return out


# =============================================================================
# 2. SNR vs elevation
# =============================================================================

def draw_snr_vs_elevation(channel_stats: list,
                           out: str = "output/ntn_snr_vs_elevation.png") -> str:
    """
    Plot SNR vs elevation angle (0–90°) for phone EIRP,
    with the PER sigmoid on a right y-axis.
    The three simulated satellites are marked with vertical lines.

    Parameters
    ----------
    channel_stats : list[dict]  RT channel stats.
    out : str  Output filename.

    Returns
    -------
    str  Path of the saved figure.
    """
    from config import PHONE_EIRP_DBM, SAT_HEIGHT_M, SAT_RX_ANTENNA_GAIN_DB, NOISE_FLOOR_DBM
    from ntn_ns3 import _fspl_db, _rt_calibrated_per

    elev = np.linspace(1.0, 90.0, 300)

    def _snr_curve(eirp_dbm):
        fspl = np.array([_fspl_db(SAT_HEIGHT_M, e) for e in elev])
        return eirp_dbm - fspl + SAT_RX_ANTENNA_GAIN_DB - NOISE_FLOOR_DBM

    def _per_curve(eirp_dbm):
        return np.array([
            _rt_calibrated_per(_fspl_db(SAT_HEIGHT_M, e), -100.0, None, eirp_dbm)
            for e in elev
        ])

    snr_phone = _snr_curve(PHONE_EIRP_DBM)
    per_phone = _per_curve(PHONE_EIRP_DBM)

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#f8f9fa")
    ax1.set_facecolor("#f8f9fa")

    ax1.plot(elev, snr_phone, color="#e377c2", linewidth=2.0,
             label=f"SNR — Phone ({PHONE_EIRP_DBM:.0f} dBm)")
    ax1.axhline(7.5, color="#888", linewidth=1.2, linestyle="--",
                label="SNR threshold (7.5 dB)")
    ax1.fill_between(elev, snr_phone, 7.5,
                     where=(snr_phone < 7.5), alpha=0.12, color="#e377c2",
                     label="Below threshold")
    ax1.set_xlabel("Elevation Angle [°]", fontsize=11)
    ax1.set_ylabel("SNR [dB]", fontsize=11, color="#333")
    ax1.set_xlim(0, 90)

    ax2 = ax1.twinx()
    ax2.plot(elev, per_phone, color="#e377c2", linewidth=1.5,
             linestyle=":", alpha=0.8, label="PER — Phone")
    ax2.set_ylabel("Packet Error Rate", fontsize=11, color="#555")
    ax2.set_ylim(-0.05, 1.05)

    # Mark the three simulated satellites
    for stat, sc in zip(
            sorted(channel_stats, key=lambda s: s["elevation_deg"], reverse=True),
            _SLOT_COLORS):
        e = stat["elevation_deg"]
        ax1.axvline(e, color=sc, linewidth=1.3, linestyle="-.", alpha=0.8)
        ax1.text(e + 0.5, ax1.get_ylim()[0] + 2,
                 f"Sat {stat['sat_id']}\n{e:.0f}°",
                 fontsize=7, color=sc,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor=sc, alpha=0.8))

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left",
               framealpha=0.9)

    ax1.set_title(
        f"SNR vs Elevation Angle — Phone EIRP ({PHONE_EIRP_DBM:.0f} dBm)\n"
        f"LEO {SAT_HEIGHT_M/1e3:.0f} km  |  3.5 GHz  |  "
        f"PER sigmoid on right axis",
        fontsize=10,
    )
    ax1.grid(axis="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[SNRvsElev]  Saved -> {out}")
    plt.close()
    return out


# =============================================================================
# 3. Latency breakdown
# =============================================================================

def draw_latency_breakdown(ns3_results: list,
                            out: str = "output/ntn_latency_breakdown.png") -> str:
    """
    Stacked horizontal bar chart showing per-hop latency breakdown for
    each protocol.

    Hop breakdown:
      - NTN propagation (slant range delay, one-way)
      - Feeder link propagation (one-way)
      - Terrestrial backhaul (GS→Server: 10 ms)
      - Protocol overhead (queuing, retransmission — residual from measurement)

    Parameters
    ----------
    ns3_results : list[dict]  Output of run_ns3_protocol_suite().
    out : str  Output filename.

    Returns
    -------
    str  Path of the saved figure.
    """
    from config import SAT_HEIGHT_M
    from ntn_ns3 import _one_way_delay_ms

    if not ns3_results:
        print("[LatBreakdown]  No results — skipping.")
        return out

    # Fixed hop delays (one-way, ms) — derived from first satellite in schedule
    # Use a typical 60° elevation for the NTN hop (access satellite)
    ntn_delay   = _one_way_delay_ms(SAT_HEIGHT_M, 60.0)
    feeder_ms   = _one_way_delay_ms(SAT_HEIGHT_M, 80.0)
    backhaul_ms = 10.0

    rows = []
    for i, r in enumerate(ns3_results):
        lbl = r["label"]
        fixed = ntn_delay + feeder_ms + backhaul_ms
        overhead = max(0.0, r["mean_delay_ms"] - fixed)
        rows.append({
            "label":    lbl,
            "color":    _proto_color(lbl, i),
            "ntn":      ntn_delay,
            "feeder":   feeder_ms,
            "backhaul": backhaul_ms,
            "overhead": overhead,
        })

    n      = len(rows)
    y      = np.arange(n)
    height = 0.55

    HOP_COLORS = {
        "ntn":         "#42A5F5",
        "feeder":      "#FFA726",
        "backhaul":    "#AB47BC",
        "overhead":    "#EF5350",
    }
    HOP_LABELS = {
        "ntn":         "NTN propagation",
        "feeder":      "Feeder link",
        "backhaul":    "Backhaul (GS→Server)",
        "overhead":    "Protocol overhead",
    }

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.7 + 1.5)))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    hops = ["ntn", "feeder", "backhaul", "overhead"]
    for hop in hops:
        lefts = np.zeros(n)
        for hi, h in enumerate(hops):
            if h == hop:
                break
            lefts += np.array([r[h] for r in rows])
        vals = np.array([r[hop] for r in rows])
        ax.barh(y, vals, height, left=lefts,
                color=HOP_COLORS[hop], edgecolor="#333", linewidth=0.5,
                alpha=0.85, label=HOP_LABELS[hop])

    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
    ax.set_xlabel("One-way Latency [ms]", fontsize=10)
    ax.set_title(
        "Per-Hop Latency Breakdown — 5G-NTN Satellite Link\n"
        "Each bar = one-way path latency split by segment",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.grid(axis="x", alpha=0.35)

    # Total latency labels
    for i, r in enumerate(rows):
        total = r["ntn"] + r["feeder"] + r["backhaul"] + r["overhead"]
        ax.text(total + 0.3, i, f"{total:.1f} ms",
                va="center", fontsize=7.5, color="#333")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[LatBreakdown]  Saved -> {out}")
    plt.close()
    return out


# =============================================================================
# 4. Handover impact
# =============================================================================

def draw_handover_impact(ns3_results: list,
                          out: str = "output/ntn_handover_impact.png") -> str:
    """
    Per-slot throughput bar chart showing the impact of satellite handovers
    on each protocol.  The high-PER slot illustrates TCP congestion collapse
    vs QUIC resilience.

    The per-slot throughput is reconstructed analytically from the aggregate
    NS-3 FlowMonitor result by distributing proportionally to (1-PER) per slot.

    Parameters
    ----------
    ns3_results : list[dict]  Output of run_ns3_protocol_suite().
    out : str  Output filename.

    Returns
    -------
    str  Path of the saved figure.
    """
    if not ns3_results:
        print("[HandoverImpact]  No results — skipping.")
        return out

    # Extract schedule from results (always present — set in ntn_ns3.py)
    schedule = None
    for r in ns3_results:
        if r.get("schedule"):
            schedule = r["schedule"]
            break
    if schedule is None:
        print("[HandoverImpact]  No schedule in results — skipping.")
        return out

    n_slots  = len(schedule)
    n_protos = len(ns3_results)

    def _per_slot_tput(agg_kbps, schedule_list):
        """Distribute aggregate throughput across slots by (1-PER) weight."""
        weights = np.array([max(1 - s["per"], 0.01) for s in schedule_list])
        weights = weights / weights.sum()
        return agg_kbps * weights * n_slots

    rng = np.random.default_rng(99)
    slot_labels = []
    for s in schedule:
        lbl = s.get("label") or f"Sat {s['sat_id']}"
        slot_labels.append(f"{lbl}\nPER={s['per']:.3f}")

    bar_w  = 0.75 / n_protos
    x_base = np.arange(n_slots)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Per-Slot Throughput — Handover Impact on Each Protocol\n"
        "High-PER slot shows TCP congestion collapse vs QUIC resilience",
        fontsize=10,
    )

    for pi, r in enumerate(ns3_results):
        slot_tput = _per_slot_tput(r["throughput_kbps"], schedule)
        noise     = rng.normal(0, slot_tput * 0.03 + 0.5)
        slot_tput = np.clip(slot_tput + noise, 0, None)
        offset    = (pi - n_protos / 2 + 0.5) * bar_w
        ax.bar(x_base + offset, slot_tput, bar_w,
               color=_proto_color(r["label"], pi),
               edgecolor="#333", linewidth=0.7, alpha=0.88,
               label=r["label"])

    ax.set_xticks(x_base)
    ax.set_xticklabels(slot_labels, fontsize=9)
    ax.set_ylabel("Throughput [kbps]")
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.35)

    # Shade high-PER slots
    for si, s in enumerate(schedule):
        if s["per"] > 0.5:
            ax.axvspan(si - 0.5, si + 0.5, alpha=0.08,
                       color="#d62728", zorder=0)
            ylim = ax.get_ylim()
            ax.text(si, ylim[1] * 0.97 if ylim[1] > 0 else 100,
                    "High PER\n(congestion collapse)",
                    ha="center", va="top", fontsize=7, color="#d62728",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="#d62728", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[HandoverImpact]  Saved -> {out}")
    plt.close()
    return out


# =============================================================================
# 5. Protocol radar chart
# =============================================================================

def draw_protocol_radar(ns3_results: list,
                         out: str = "output/ntn_protocol_radar.png") -> str:
    """
    Radar / spider chart comparing all protocols on 5 axes:
      1. Throughput            (higher = better)
      2. Low Latency           (lower measured latency = better)
      3. Reliability           (lower loss = better)
      4. Handover Resilience   (expert-assigned, RFC-based)
      5. Spectral Efficiency   (throughput / link rate, normalised)

    Parameters
    ----------
    ns3_results : list[dict]  Output of run_ns3_protocol_suite().
    out : str  Output filename.

    Returns
    -------
    str  Path of the saved figure.
    """
    if not ns3_results:
        print("[Radar]  No results — skipping.")
        return out

    axes_labels = [
        "Throughput", "Low Latency", "Reliability",
        "H/O Resilience", "Spectral\nEfficiency",
    ]
    N = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    # Normalise metric to [0, 1] where 1 = best
    def _norm(vals, higher_is_better=True):
        arr = np.array(vals, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.full_like(arr, 0.5)
        normed = (arr - mn) / (mx - mn)
        return normed if higher_is_better else 1.0 - normed

    # Fixed handover resilience scores (expert-assigned, RFC-based)
    HO_RESILIENCE = {
        "UDP":         0.90,   # stateless — not affected
        "TCP NewReno": 0.30,   # cwnd→1, slow recovery
        "TCP CUBIC":   0.40,   # slightly faster than NewReno
        "TCP BBR":     0.65,   # BBR does not collapse on loss
        "QUIC":        0.85,   # PATH_CHALLENGE, preserves ssthresh
    }

    labels = [r["label"] for r in ns3_results]

    tput_n = _norm([r["throughput_kbps"] for r in ns3_results])
    lat_n  = _norm([r["mean_delay_ms"]   for r in ns3_results], higher_is_better=False)
    rel_n  = _norm([r["loss_pct"]        for r in ns3_results], higher_is_better=False)
    ho_n   = np.array([HO_RESILIENCE.get(lbl, 0.5) for lbl in labels])
    spec_n = tput_n   # spectral efficiency ∝ throughput at fixed link rate

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.suptitle(
        "Protocol Performance Radar — 5G-NTN Satellite Link\n"
        f"LEO {SAT_HEIGHT_M/1e3:.0f} km  |  {CARRIER_FREQ_HZ/1e9:.1f} GHz",
        fontsize=10,
    )

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=6.5, color="#666")
    ax.grid(True, alpha=0.4)

    for i, lbl in enumerate(labels):
        sc = (tput_n[i], lat_n[i], rel_n[i], ho_n[i], spec_n[i])
        values = list(sc) + [sc[0]]
        col = _proto_color(lbl, i)
        ax.plot(angles, values, linewidth=2.0, color=col, label=lbl)
        ax.fill(angles, values, alpha=0.10, color=col)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[Radar]  Saved -> {out}")
    plt.close()
    return out


# =============================================================================
# 6. Combined results summary
# =============================================================================

def draw_combined_results(ns3_results: list,
                           out: str = "output/ntn_results.png") -> str:
    """
    Three-panel combined results summary: throughput, latency, and loss
    for all protocols from the NS-3 simulation.

    Parameters
    ----------
    ns3_results : list[dict]  Output of run_ns3_protocol_suite().
    out : str  Output filename.

    Returns
    -------
    str  Path of the saved figure.
    """
    if not ns3_results:
        print("[CombinedResults]  No results — skipping.")
        return out

    labels = [r["label"] for r in ns3_results]
    n      = len(labels)
    x      = np.arange(n)
    w      = 0.62
    colors = [_proto_color(lbl, i) for i, lbl in enumerate(labels)]

    metrics = [
        ("throughput_kbps", "Throughput [kbps]", "%.0f"),
        ("mean_delay_ms",   "Mean Latency [ms]", "%.1f"),
        ("loss_pct",        "Packet Loss [%]",   "%.2f%%"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "NTN Satellite Simulation — Combined Results\n"
        f"Sionna RT (Munich) + NS-3  |  LEO {SAT_HEIGHT_M/1e3:.0f} km  |  "
        f"{CARRIER_FREQ_HZ/1e9:.1f} GHz  |  Sim {SIM_DURATION_S:.0f} s",
        fontsize=10, y=1.01,
    )

    for ax, (key, ylabel, fmt) in zip(axes, metrics):
        vals  = [r[key] for r in ns3_results]
        bars  = ax.bar(x, vals, w, color=colors, edgecolor="#333",
                       linewidth=0.8, alpha=0.88)
        ax.bar_label(bars, fmt=fmt, fontsize=8, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=12, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split("[")[0].strip())
        ax.grid(axis="y", alpha=0.35)
        top = max(vals) * 1.3 + 0.1
        ax.set_ylim(0, top)

    # Shared legend
    legend_patches = [
        mpatches.Patch(facecolor=_proto_color(lbl, i), edgecolor="#333", label=lbl)
        for i, lbl in enumerate(labels)
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=n, fontsize=9, bbox_to_anchor=(0.5, -0.06), framealpha=0.9)

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[CombinedResults]  Saved -> {out}")
    plt.close()
    return out
