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


# =============================================================================
# Throughput-over-time graph
# =============================================================================

# Handover schedule constants (derived from RT + NS-3 simulation)
_SCHEDULE = [
    {"label": "Sat 0", "t_start": 0,  "t_end": 20, "per": 0.101,
     "elev": 69.9, "delay_ms": 2.1, "color": "#1a6fc4"},
    {"label": "Sat 1", "t_start": 20, "t_end": 40, "per": 0.033,
     "elev": 54.9, "delay_ms": 2.4, "color": "#2ca02c"},
    {"label": "Sat 2", "t_start": 40, "t_end": 60, "per": 0.768,
     "elev": 39.9, "delay_ms": 3.0, "color": "#d62728"},
]

_UDP_RATE_KBPS = 5000.0   # CBR source rate


def _slot_per(t: np.ndarray) -> np.ndarray:
    """Return per-sample PER value based on which satellite is active."""
    per = np.full_like(t, _SCHEDULE[0]["per"])
    for slot in _SCHEDULE[1:]:
        per[t >= slot["t_start"]] = slot["per"]
    return per


def draw_throughput_over_time(
        ns3_results: list | None = None,
        out: str = "ntn_throughput_over_time.png") -> str:
    """
    Plot analytically reconstructed per-protocol throughput vs. time,
    with shaded background regions and vertical dashed lines marking each
    satellite handover event.

    Since NS-3 FlowMonitor only provides aggregate flow statistics, the
    time-series is reconstructed analytically:
      - Per-slot PER values come from the RT-calibrated link budget.
      - UDP throughput = CBR_rate × (1 - PER) + Gaussian noise.
      - TCP throughput is derived from the aggregate NS-3 result,
        split proportionally across slots by their relative (1-PER) weight,
        then smoothed with a rolling average to mimic congestion-window
        growth and brief post-handover recovery dip.

    Parameters
    ----------
    ns3_results : list[dict] | None
        Output of run_ns3_protocol_suite().  If None, placeholder values
        derived from the known simulation run are used so the plot can be
        generated without re-running NS-3.
    out : str
        Output filename.

    Returns
    -------
    str  Path of the saved figure.
    """
    rng = np.random.default_rng(42)

    dt   = 0.1   # seconds per sample
    t    = np.arange(0, SIM_DURATION_S + dt, dt)
    per  = _slot_per(t)

    # ── Protocol aggregate targets (from NS-3 run; fallback if no results) ────
    _defaults = {
        "UDP":         4830.0,
        "TCP NewReno":  382.0,
        "TCP CUBIC":    618.0,
        "TCP BBR":     4185.0,
    }
    if ns3_results:
        agg = {r["label"]: r["throughput_kbps"] for r in ns3_results}
    else:
        agg = _defaults

    # ── Reconstruct time-series per protocol ──────────────────────────────────
    def _udp_trace() -> np.ndarray:
        base   = _UDP_RATE_KBPS * (1.0 - per)
        noise  = rng.normal(0, 80, size=len(t))
        return np.clip(base + noise, 0, _UDP_RATE_KBPS)

    def _tcp_trace(label: str) -> np.ndarray:
        # Slot weights proportional to relative link quality (1-PER)
        weights = np.array([1 - s["per"] for s in _SCHEDULE])
        weights = weights / weights.sum()
        total   = agg.get(label, _defaults[label])

        # Per-slot mean throughput
        slot_mean = total * weights * (len(_SCHEDULE))

        # Build base trace
        base = np.zeros(len(t))
        for i, slot in enumerate(_SCHEDULE):
            mask = (t >= slot["t_start"]) & (t < slot["t_end"])
            base[mask] = slot_mean[i]

        # Add noise
        noise = rng.normal(0, total * 0.04, size=len(t))
        trace = np.clip(base + noise, 0, None)

        # Model post-handover recovery dip (CWND reset at handover boundaries)
        ho_times = [20.0, 40.0]
        recovery_s = 4.0  # seconds to recover
        for ho in ho_times:
            mask = (t >= ho) & (t < ho + recovery_s)
            dip  = 1.0 - 0.55 * np.exp(-(t[mask] - ho) / (recovery_s * 0.4))
            trace[mask] *= dip

        # Smooth with rolling average (~3 s window)
        win = max(1, int(3.0 / dt))
        kernel = np.ones(win) / win
        trace = np.convolve(trace, kernel, mode="same")
        return np.clip(trace, 0, None)

    traces = {
        "UDP":         _udp_trace(),
        "TCP NewReno": _tcp_trace("TCP NewReno"),
        "TCP CUBIC":   _tcp_trace("TCP CUBIC"),
        "TCP BBR":     _tcp_trace("TCP BBR"),
    }

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    # Shaded background regions per satellite slot — draw after ylim is fixed
    slot_alphas = [0.10, 0.07, 0.13]
    for slot, alpha in zip(_SCHEDULE, slot_alphas):
        ax.axvspan(slot["t_start"], slot["t_end"],
                   color=slot["color"], alpha=alpha, zorder=0)

    # Handover vertical lines
    for ho_t in [20.0, 40.0]:
        ax.axvline(ho_t, color="#333", linestyle="--", linewidth=1.4,
                   zorder=3, label="_nolegend_")
        ax.text(ho_t + 0.4, 50,
                f"Handover\nt = {ho_t:.0f} s",
                fontsize=7, color="#333", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#333", alpha=0.8))

    # Protocol traces
    for label, trace in traces.items():
        ax.plot(t, trace, color=PROTO_COLORS.get(label, "gray"),
                linewidth=1.8, label=label, alpha=0.9)

    ax.set_xlabel("Simulation Time [s]", fontsize=11)
    ax.set_ylabel("Throughput [kbps]", fontsize=11)
    ax.set_title(
        "Transport Protocol Throughput over Time — 5G-NTN LEO Satellite Link\n"
        "Analytically reconstructed from NS-3 aggregate stats + RT-calibrated PER",
        fontsize=10,
    )
    ax.set_xlim(0, SIM_DURATION_S)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.35)
    ax.grid(axis="x", alpha=0.2)

    # Slot labels at top of plot (drawn after ylim is finalised)
    y_max = ax.get_ylim()[1]
    for slot in _SCHEDULE:
        mid = (slot["t_start"] + slot["t_end"]) / 2
        ax.text(mid, y_max * 0.97,
                f"{slot['label']}\nelev {slot['elev']:.0f}°\nPER {slot['per']:.3f}",
                ha="center", va="top", fontsize=7.5, color=slot["color"],
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=slot["color"], alpha=0.75), zorder=4)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[ThroughputTime]  Saved -> {out}")
    plt.close()
    return out


# =============================================================================
# Detailed network illustration
# =============================================================================

def draw_network_illustration(out: str = "ntn_network_illustration.png") -> str:
    """
    Draw a detailed end-to-end NTN network diagram including:
      - Earth arc at the bottom with ground infrastructure
      - LEO orbit arc with 3 satellites
      - Service link, feeder link, and fibre link with parameter labels
      - Protocol stack labels on relevant nodes
      - Link delay, data rate, and PER annotations

    Returns
    -------
    str  Path of the saved figure.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    # ── Earth arc ─────────────────────────────────────────────────────────────
    earth_cx, earth_cy = 8.0, -7.5
    earth_r = 10.5
    earth_arc = mpatches.Arc(
        (earth_cx, earth_cy), 2 * earth_r, 2 * earth_r,
        angle=0, theta1=55, theta2=125,
        color="#3a7d44", linewidth=3, zorder=2,
    )
    ax.add_patch(earth_arc)

    # Filled Earth surface hint
    from matplotlib.patches import Wedge
    earth_fill = Wedge(
        (earth_cx, earth_cy), earth_r,
        theta1=55, theta2=125,
        facecolor="#1a4a1f", edgecolor="#3a7d44", linewidth=2, zorder=1,
    )
    ax.add_patch(earth_fill)

    # Atmosphere haze
    atmos = mpatches.Arc(
        (earth_cx, earth_cy), 2 * (earth_r + 0.35), 2 * (earth_r + 0.35),
        angle=0, theta1=55, theta2=125,
        color="#4fc3f7", linewidth=1.0, linestyle=":", alpha=0.5, zorder=2,
    )
    ax.add_patch(atmos)

    # ── Orbit arc ─────────────────────────────────────────────────────────────
    orbit_r = earth_r + 2.8
    orbit_arc = mpatches.Arc(
        (earth_cx, earth_cy), 2 * orbit_r, 2 * orbit_r,
        angle=0, theta1=57, theta2=123,
        color="#aaaaaa", linewidth=1.2, linestyle="--", alpha=0.6, zorder=2,
    )
    ax.add_patch(orbit_arc)
    # Orbit label
    import math
    orb_label_angle = math.radians(90)
    ax.text(
        earth_cx + (orbit_r + 0.3) * math.cos(orb_label_angle),
        earth_cy + (orbit_r + 0.3) * math.sin(orb_label_angle),
        "LEO orbit  ~600 km",
        ha="center", va="bottom", fontsize=7.5, color="#aaaaaa",
        fontstyle="italic",
    )

    # ── Satellite positions ────────────────────────────────────────────────────
    # Sat 0: left (70° elevation → angle 110° from Earth center on our arc)
    # Sat 1: centre (55° elevation → angle 90°)
    # Sat 2: right (40° elevation → angle 70°)
    sat_angles_deg = [110, 90, 70]
    sat_elevs      = [69.9, 54.9, 39.9]
    sat_pers       = [0.101, 0.033, 0.768]
    sat_delays     = [2.1, 2.4, 3.0]
    sat_colors     = [_SCHEDULE[0]["color"], _SCHEDULE[1]["color"],
                      _SCHEDULE[2]["color"]]
    sat_positions  = []

    for i, (ang, elev, per, dly, sc) in enumerate(
            zip(sat_angles_deg, sat_elevs, sat_pers, sat_delays, sat_colors)):
        rad = math.radians(ang)
        sx  = earth_cx + orbit_r * math.cos(rad)
        sy  = earth_cy + orbit_r * math.sin(rad)
        sat_positions.append((sx, sy))

        # Satellite body (diamond)
        diamond = plt.Polygon(
            [[sx, sy + 0.22], [sx + 0.18, sy], [sx, sy - 0.22],
             [sx - 0.18, sy]],
            closed=True, facecolor=sc, edgecolor="white",
            linewidth=1.2, zorder=5,
        )
        ax.add_patch(diamond)

        # Solar panel wings
        ax.plot([sx - 0.45, sx - 0.18], [sy, sy], color=sc,
                linewidth=2.5, solid_capstyle="round", zorder=4)
        ax.plot([sx + 0.18, sx + 0.45], [sy, sy], color=sc,
                linewidth=2.5, solid_capstyle="round", zorder=4)

        # Label box
        lbl_dx = (-1.2 if i == 0 else (0.0 if i == 1 else 1.2))
        lbl_dy = 0.55
        ax.text(sx + lbl_dx, sy + lbl_dy,
                f"Sat {i}  (elev {elev:.0f}°)\nPER={per:.3f}  dly={dly:.1f} ms",
                ha="center", va="bottom", fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#1a2a3a",
                          edgecolor=sc, linewidth=1.0),
                zorder=6)

    # ── Ground nodes ──────────────────────────────────────────────────────────
    def _gnd_box(cx, cy, lines, fc="#1c3a5e", ec="#4a9fd4", fs=8):
        n_lines = len(lines)
        h = 0.15 * n_lines + 0.25
        w = max(len(l) for l in lines) * 0.085 + 0.3
        box = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.06",
            facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=5,
        )
        ax.add_patch(box)
        for j, line in enumerate(lines):
            yoff = (n_lines - 1) / 2 * 0.15 - j * 0.15
            ax.text(cx, cy + yoff, line, ha="center", va="center",
                    fontsize=fs, color="white", zorder=6,
                    fontweight=("bold" if j == 0 else "normal"))

    # UE on the left side of the Earth arc
    ue_rad = math.radians(108)
    ue_sx  = earth_cx + earth_r * math.cos(ue_rad)
    ue_sy  = earth_cy + earth_r * math.sin(ue_rad)
    _gnd_box(ue_sx, ue_sy + 0.35,
             ["UE (Phone)", "[50, 80, 1.5 m]",
              "5G NR NTN UE", "3.5 GHz"],
             fc="#1a3a1a", ec="#3a7d44")

    # Ground station in the centre
    gs_rad = math.radians(90)
    gs_sx  = earth_cx + earth_r * math.cos(gs_rad)
    gs_sy  = earth_cy + earth_r * math.sin(gs_rad)
    _gnd_box(gs_sx, gs_sy + 0.35,
             ["Ground Station", "Ka-band gateway",
              "100 Mbps uplink"],
             fc="#3a2a00", ec="#f0a500")

    # Internet server on the right
    srv_rad = math.radians(72)
    srv_sx  = earth_cx + earth_r * math.cos(srv_rad)
    srv_sy  = earth_cy + earth_r * math.sin(srv_rad)
    _gnd_box(srv_sx, srv_sy + 0.35,
             ["Internet Server", "TCP/UDP sink",
              "NS-3 PacketSink"],
             fc="#3a001a", ec="#e05080")

    # ── Links ─────────────────────────────────────────────────────────────────
    def _link(x1, y1, x2, y2, label, color="#aaddff",
              ls="-", lw=1.5, label_offset=(0, 0)):
        ax.plot([x1, x2], [y1, y2], color=color, linestyle=ls,
                linewidth=lw, zorder=3, alpha=0.85)
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=6.5, color=color, zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#0d1b2a",
                          edgecolor="none", alpha=0.85))

    ue_top  = (ue_sx, ue_sy + 0.35 + 0.25)
    gs_top  = (gs_sx, gs_sy + 0.35 + 0.25)
    srv_top = (srv_sx, srv_sy + 0.35 + 0.25)

    # UE → Sat 0 (active service link)
    _link(ue_top[0], ue_top[1], sat_positions[0][0], sat_positions[0][1] - 0.22,
          "5G-NR NTN service link\n3.5 GHz  |  dly 2.1 ms  |  PER 0.101",
          color="#64b5f6", lw=2.2, label_offset=(-0.6, 0.2))

    # UE → Sat 1 (handover candidate, dashed)
    _link(ue_top[0], ue_top[1], sat_positions[1][0], sat_positions[1][1] - 0.22,
          "handover candidate\nPER 0.033",
          color="#80cbc4", ls="--", lw=1.3, label_offset=(0.1, 0.3))

    # UE → Sat 2 (future, dotted)
    _link(ue_top[0], ue_top[1], sat_positions[2][0], sat_positions[2][1] - 0.22,
          "future\nPER 0.768",
          color="#ef9a9a", ls=":", lw=1.0, label_offset=(0.6, 0.1))

    # Sat 0 → GS (feeder link)
    _link(sat_positions[0][0], sat_positions[0][1] - 0.22,
          gs_top[0], gs_top[1],
          "Ka-band feeder link\n26.5 GHz  |  100 Mbps",
          color="#ffcc80", lw=1.8, label_offset=(-0.3, 0.1))

    # GS → Server (fibre)
    _link(gs_top[0], gs_top[1], srv_top[0], srv_top[1],
          "Fibre  1 Gbps  |  10 ms",
          color="#a5d6a7", lw=1.8, label_offset=(0, 0.2))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        plt.Line2D([0], [0], color="#64b5f6", lw=2.2, label="Active service link"),
        plt.Line2D([0], [0], color="#80cbc4", lw=1.3,
                   linestyle="--", label="Handover candidate"),
        plt.Line2D([0], [0], color="#ef9a9a", lw=1.0,
                   linestyle=":", label="Future satellite"),
        plt.Line2D([0], [0], color="#ffcc80", lw=1.8,
                   label="Ka-band feeder link"),
        plt.Line2D([0], [0], color="#a5d6a7", lw=1.8,
                   label="Fibre backhaul"),
        plt.Line2D([0], [0], color="#aaaaaa", lw=1.2,
                   linestyle="--", label="LEO orbit arc"),
    ]
    ax.legend(handles=legend_items, loc="upper left",
              fontsize=7.5, framealpha=0.85,
              facecolor="#1a2a3a", labelcolor="white",
              edgecolor="#4a9fd4")

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        "5G Non-Terrestrial Network (NTN) — End-to-End Topology\n"
        f"LEO {SAT_HEIGHT_M/1e3:.0f} km  |  {CARRIER_FREQ_HZ/1e9:.1f} GHz  |  "
        f"3-satellite constellation  |  2 handover events",
        fontsize=11, color="white", pad=10,
    )

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[NetworkIllust]  Saved -> {out}")
    plt.close()
    return out


# =============================================================================
# UE-to-satellite street scene
# =============================================================================

def draw_ue_satellite_scene(out: str = "ntn_ue_satellite.png") -> str:
    """
    Draw an artistic street-level scene illustrating the UE-to-satellite
    communication geometry with multipath rays, as derived from Sionna RT.

    Scene elements:
      - Night sky gradient background
      - Two city building silhouettes forming a street canyon
      - UE (smartphone) at street level
      - LEO satellite high above
      - Line-of-sight ray (direct path)
      - Two reflected rays bouncing off building walls
      - Signal strength annotation and elevation arc

    Returns
    -------
    str  Path of the saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # ── Sky gradient ──────────────────────────────────────────────────────────
    from matplotlib.colors import LinearSegmentedColormap
    sky_cmap = LinearSegmentedColormap.from_list(
        "sky", ["#0d1b2a", "#1a3a5c", "#2a5a8c"])
    sky_grad = np.linspace(0, 1, 256).reshape(256, 1)
    ax.imshow(sky_grad, aspect="auto", cmap=sky_cmap,
              extent=[0, 10, 0, 12], origin="lower",
              zorder=0, alpha=0.95)

    # Stars (scattered dots)
    rng = np.random.default_rng(7)
    star_x = rng.uniform(0.2, 9.8, 60)
    star_y = rng.uniform(5.5, 11.8, 60)
    star_s = rng.uniform(2, 10, 60)
    ax.scatter(star_x, star_y, s=star_s, color="white", alpha=0.6, zorder=1)

    # ── Ground / street ───────────────────────────────────────────────────────
    street = plt.Polygon([[0, 0], [10, 0], [10, 1.2], [0, 1.2]],
                         closed=True, facecolor="#1c1c1c", edgecolor="none",
                         zorder=2)
    ax.add_patch(street)
    # Road markings
    for xm in np.arange(1.0, 9.5, 1.5):
        ax.plot([xm, xm + 0.8], [0.6, 0.6], color="#ffdd44",
                linewidth=2.5, alpha=0.6, zorder=3)

    # ── Buildings ─────────────────────────────────────────────────────────────
    def _building(x, w, h, fc="#2a3a4a", window_color="#ffdd88"):
        body = plt.Polygon(
            [[x, 1.2], [x + w, 1.2], [x + w, 1.2 + h], [x, 1.2 + h]],
            closed=True, facecolor=fc, edgecolor="#4a5a6a",
            linewidth=0.8, zorder=2,
        )
        ax.add_patch(body)
        # Windows
        ww, wh = 0.22, 0.28
        for row in range(int(h / 0.7)):
            for col in range(int(w / 0.55)):
                wx = x + 0.18 + col * 0.55
                wy = 1.2 + 0.25 + row * 0.7
                if wx + ww < x + w - 0.1 and wy + wh < 1.2 + h - 0.1:
                    lit = rng.random() > 0.35
                    win = plt.Polygon(
                        [[wx, wy], [wx + ww, wy],
                         [wx + ww, wy + wh], [wx, wy + wh]],
                        closed=True,
                        facecolor=window_color if lit else "#1a2030",
                        edgecolor="#3a4a5a", linewidth=0.5, zorder=3,
                    )
                    ax.add_patch(win)

    _building(0.0, 2.8, 5.2, fc="#253040")   # Left building
    _building(7.2, 2.8, 4.0, fc="#253545")   # Right building

    # ── UE (phone icon) ───────────────────────────────────────────────────────
    ue_x, ue_y = 5.0, 1.55
    phone = FancyBboxPatch(
        (ue_x - 0.18, ue_y - 0.42), 0.36, 0.74,
        boxstyle="round,pad=0.04",
        facecolor="#222", edgecolor="#aaaaaa", linewidth=1.5, zorder=8,
    )
    ax.add_patch(phone)
    # Screen
    screen = FancyBboxPatch(
        (ue_x - 0.13, ue_y - 0.30), 0.26, 0.48,
        boxstyle="round,pad=0.02",
        facecolor="#1a6fc4", edgecolor="none", linewidth=0, zorder=9,
    )
    ax.add_patch(screen)
    # Home button
    ax.add_patch(plt.Circle((ue_x, ue_y - 0.36), 0.04,
                             color="#555", zorder=9))
    ax.text(ue_x, ue_y - 0.68, "UE\n[50, 80, 1.5 m]",
            ha="center", va="top", fontsize=7.5, color="white", zorder=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0d1b2a",
                      edgecolor="#4a9fd4", alpha=0.8))

    # ── Satellite ─────────────────────────────────────────────────────────────
    sat_x, sat_y = 6.8, 10.5
    # Body
    sat_body = plt.Polygon(
        [[sat_x, sat_y + 0.35], [sat_x + 0.28, sat_y],
         [sat_x, sat_y - 0.35], [sat_x - 0.28, sat_y]],
        closed=True, facecolor="#2ca02c", edgecolor="white",
        linewidth=1.5, zorder=8,
    )
    ax.add_patch(sat_body)
    # Solar panels
    for dx in [(-0.72, -0.28), (0.28, 0.72)]:
        ax.plot([sat_x + dx[0], sat_x + dx[1]], [sat_y, sat_y],
                color="#ffaa00", linewidth=5, solid_capstyle="butt",
                zorder=7, alpha=0.9)
    ax.text(sat_x + 0.5, sat_y + 0.5,
            "LEO Sat 1\nelev 55°  |  PER 0.033",
            ha="left", va="bottom", fontsize=7.5, color="white", zorder=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0d1b2a",
                      edgecolor="#2ca02c", alpha=0.85))

    # ── Ray paths (Sionna RT multipath) ──────────────────────────────────────
    # LoS ray (direct path, brightest)
    ax.annotate("", xy=(sat_x - 0.2, sat_y - 0.35),
                xytext=(ue_x + 0.05, ue_y + 0.32),
                arrowprops=dict(arrowstyle="-|>", color="#64b5f6",
                                lw=2.0, mutation_scale=12),
                zorder=6)
    ax.text(5.75, 6.5, "LoS path\n(direct)",
            ha="center", fontsize=7, color="#64b5f6",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0d1b2a",
                      edgecolor="#64b5f6", alpha=0.8), zorder=7)

    # Reflected ray 1: off left building wall → UE
    refl1_x, refl1_y = 2.8, 5.8   # reflection point on left wall
    # Sat → wall
    ax.annotate("", xy=(refl1_x, refl1_y),
                xytext=(sat_x - 0.28, sat_y),
                arrowprops=dict(arrowstyle="-", color="#ffab40",
                                lw=1.5, linestyle="dashed"),
                zorder=5)
    # Wall → UE
    ax.annotate("", xy=(ue_x - 0.18, ue_y + 0.1),
                xytext=(refl1_x, refl1_y),
                arrowprops=dict(arrowstyle="-|>", color="#ffab40",
                                lw=1.5, mutation_scale=10),
                zorder=5)
    # Reflection marker
    ax.plot(refl1_x, refl1_y, "o", ms=6, color="#ffab40", zorder=6)
    ax.text(refl1_x - 0.35, refl1_y,
            "Refl. 1\n(left wall)",
            ha="right", fontsize=6.5, color="#ffab40",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#0d1b2a",
                      edgecolor="#ffab40", alpha=0.8), zorder=7)

    # Reflected ray 2: off right building wall → UE
    refl2_x, refl2_y = 7.2, 4.5   # reflection point on right wall
    ax.annotate("", xy=(refl2_x, refl2_y),
                xytext=(sat_x - 0.05, sat_y - 0.35),
                arrowprops=dict(arrowstyle="-", color="#ce93d8",
                                lw=1.5, linestyle="dashed"),
                zorder=5)
    ax.annotate("", xy=(ue_x + 0.18, ue_y + 0.1),
                xytext=(refl2_x, refl2_y),
                arrowprops=dict(arrowstyle="-|>", color="#ce93d8",
                                lw=1.5, mutation_scale=10),
                zorder=5)
    ax.plot(refl2_x, refl2_y, "o", ms=6, color="#ce93d8", zorder=6)
    ax.text(refl2_x + 0.15, refl2_y,
            "Refl. 2\n(right wall)",
            ha="left", fontsize=6.5, color="#ce93d8",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#0d1b2a",
                      edgecolor="#ce93d8", alpha=0.8), zorder=7)

    # ── Elevation arc annotation ──────────────────────────────────────────────
    elev_arc = mpatches.Arc(
        (ue_x, ue_y + 0.32), 2.2, 2.2,
        angle=0, theta1=55, theta2=90,
        color="#80cbc4", linewidth=1.2, zorder=7,
    )
    ax.add_patch(elev_arc)
    ax.text(ue_x + 0.8, ue_y + 1.6, "55°", fontsize=8,
            color="#80cbc4", ha="center", zorder=8)

    # ── Info panel ────────────────────────────────────────────────────────────
    info = (
        "Sionna RT — Munich scene\n"
        "8 multipath components resolved\n"
        "Delay spread: ~12 ns   |   Mean path gain: −142 dB"
    )
    ax.text(5.0, 0.35, info, ha="center", va="center",
            fontsize=7.5, color="white", zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d2a1a",
                      edgecolor="#3a7d44", alpha=0.9))

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        "UE-to-Satellite Communication — Urban Street Canyon\n"
        "Multipath propagation via Sionna RT ray tracing (Munich scene)",
        fontsize=10, color="#cce4ff", pad=8,
    )
    fig.patch.set_facecolor("#0d1b2a")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[UESatScene]  Saved -> {out}")
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
