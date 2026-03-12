"""
topology_diagram.py — Scenario topology illustration and protocol comparison
=============================================================================
Produces two output figures:

  ntn_topology.png
      A hand-drawn-style diagram of the end-to-end NTN scenario:
        Phone (UE)  ──5G-NR NTN──  LEO Satellite  ──Ka feeder──
        Ground Station  ──fibre──  Internet Server
      Multiple satellites are shown with a handover arrow.

  ntn_protocol_comparison.png
      A grouped bar chart comparing the four transport protocols
      (UDP, TCP NewReno, TCP CUBIC, TCP BBR) across:
        - Mean end-to-end latency [ms]
        - Throughput [kbps]
        - Packet loss [%]

Both figures are saved and also returned so main.py can embed them into a
single multi-panel summary figure if desired.
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
}

# Fallback for any unlisted label
_FALLBACK_COLORS = ["#9467bd", "#8c564b", "#17becf", "#bcbd22"]


def _proto_color(label: str, idx: int) -> str:
    return PROTO_COLORS.get(label, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


# =============================================================================
# Topology diagram
# =============================================================================

def _draw_node(ax, cx, cy, label, sublabel=None,
               box_color="#dce9f7", text_color="black",
               width=1.4, height=0.55):
    """Draw a rounded rectangle node with a label (and optional sub-label)."""
    box = FancyBboxPatch(
        (cx - width / 2, cy - height / 2),
        width, height,
        boxstyle="round,pad=0.07",
        facecolor=box_color,
        edgecolor="#444",
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(cx, cy + (0.07 if sublabel else 0), label,
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color=text_color, zorder=4)
    if sublabel:
        ax.text(cx, cy - 0.13, sublabel,
                ha="center", va="center", fontsize=6.5,
                color="#555", zorder=4)


def _draw_arrow(ax, x1, y1, x2, y2, label="", color="#333",
                linestyle="-", lw=1.5, label_offset=(0, 0.12)):
    """Draw a simple annotated arrow between two points."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            linestyle=linestyle,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=2,
    )
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="bottom",
                fontsize=6.5, color=color, zorder=5,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])


def draw_topology(out: str = "ntn_topology.png") -> str:
    """
    Draw the end-to-end NTN topology scenario and save to *out*.

    Returns
    -------
    str  Path of the saved figure.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.8, 4.2)
    ax.axis("off")

    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    fig.suptitle(
        "5G-NTN Satellite Link Simulation — End-to-End Scenario\n"
        f"LEO {SAT_HEIGHT_M/1e3:.0f} km  |  {CARRIER_FREQ_HZ/1e9:.1f} GHz  |  "
        f"SIM_DURATION = {SIM_DURATION_S:.0f} s",
        fontsize=10, y=0.97,
    )

    # ── Nodes ─────────────────────────────────────────────────────────────────
    # Row 1 (top): satellites
    _draw_node(ax, 2.8, 3.2,  "LEO Sat 0",  "elev ≈ 90°",
               box_color="#cce5ff", width=1.5)
    _draw_node(ax, 5.5, 3.5,  "LEO Sat 1",  "elev ≈ 75°",
               box_color="#cce5ff", width=1.5)
    _draw_node(ax, 8.0, 3.1,  "LEO Sat 2",  "elev ≈ 60°",
               box_color="#cce5ff", width=1.5)

    # Row 2 (middle): UE and ground station
    _draw_node(ax, 1.5, 1.3,  "UE (Phone)", "[0, 0, 1.5 m]",
               box_color="#d4edda", width=1.5)
    _draw_node(ax, 7.5, 1.3,  "Ground\nStation", "GS, Ka-band",
               box_color="#fff3cd", width=1.5, height=0.65)

    # Row 3 (bottom): Internet server
    _draw_node(ax, 10.5, 1.3, "Internet\nServer", "TCP/UDP sink",
               box_color="#f8d7da", width=1.5, height=0.65)

    # ── UE → Satellite links (service links) ─────────────────────────────────
    # UE → Sat0 (serving)
    _draw_arrow(ax, 1.5, 1.58, 2.8, 2.93,
                label="5G-NR NTN\n(serving)",
                color="#0056b3", lw=2.0,
                label_offset=(-0.55, 0.0))

    # UE → Sat1 (next, dashed)
    _draw_arrow(ax, 1.5, 1.58, 5.5, 3.23,
                label="handover candidate",
                color="#6c757d", linestyle="--", lw=1.2,
                label_offset=(0.2, 0.05))

    # UE → Sat2 (future, dotted)
    _draw_arrow(ax, 1.5, 1.58, 8.0, 2.83,
                label="",
                color="#adb5bd", linestyle=":", lw=1.0)

    # ── Sat → Ground station (feeder link) ───────────────────────────────────
    _draw_arrow(ax, 2.8, 2.93, 7.5, 1.62,
                label="Ka feeder  100 Mbps",
                color="#856404", lw=1.5,
                label_offset=(0.1, 0.12))

    # ── Handover arrow (between satellites) ──────────────────────────────────
    ax.annotate(
        "", xy=(5.5, 3.5), xytext=(2.8, 3.2),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#dc3545",
            lw=1.5,
            connectionstyle="arc3,rad=0.25",
        ),
        zorder=2,
    )
    ax.text(4.1, 3.82, "handover\n(elev < 10°)", ha="center", va="bottom",
            fontsize=6.5, color="#dc3545",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # ── Ground station → Internet server ─────────────────────────────────────
    _draw_arrow(ax, 8.25, 1.3, 9.75, 1.3,
                label="Fibre  1 Gbps\n10 ms",
                color="#155724", lw=1.5,
                label_offset=(0, 0.14))

    # ── Protocol stack labels on UE ───────────────────────────────────────────
    ax.text(1.5, 0.45,
            "Protocols: UDP | TCP NewReno | TCP CUBIC | TCP BBR",
            ha="center", va="center", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e2e3e5",
                      edgecolor="#aaa"),
            zorder=5)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor="#cce5ff", edgecolor="#444", label="LEO Satellite"),
        mpatches.Patch(facecolor="#d4edda", edgecolor="#444", label="UE (Phone)"),
        mpatches.Patch(facecolor="#fff3cd", edgecolor="#444", label="Ground Station"),
        mpatches.Patch(facecolor="#f8d7da", edgecolor="#444", label="Internet Server"),
        plt.Line2D([0], [0], color="#0056b3", lw=2, label="Active service link"),
        plt.Line2D([0], [0], color="#6c757d", lw=1.2,
                   linestyle="--", label="Handover candidate"),
        plt.Line2D([0], [0], color="#dc3545", lw=1.5, label="Handover event"),
    ]
    ax.legend(handles=legend_elements, loc="lower left",
              fontsize=7, framealpha=0.9, ncol=2,
              bbox_to_anchor=(0.0, -0.02))

    plt.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[Topology]  Saved -> {out}")
    plt.close()
    return out


# =============================================================================
# Protocol comparison bar chart
# =============================================================================

def draw_protocol_comparison(ns3_results: list,
                              out: str = "ntn_protocol_comparison.png") -> str:
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
        f"{SIM_DURATION_S:.0f} s, {len(ns3_results[0].get('label', '')) and ''}"
        f"Munich urban scene)",
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
    # Show a small info row at the bottom with per-protocol service-link stats
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
                 out: str = "ntn_summary.png") -> str:
    """
    Five-panel summary figure combining BER/BLER (Part 1) with the
    protocol comparison bar charts (Part 2) in a single page.

    Parameters
    ----------
    ber_results  : dict  Mapping scenario -> (ber_array, bler_array).
    ns3_results  : list  Output of run_ns3_protocol_suite().
    out          : str   Output filename.

    Returns
    -------
    str  Path of the saved figure.
    """
    snr_range = np.arange(0, 22, 2, dtype=float)
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
