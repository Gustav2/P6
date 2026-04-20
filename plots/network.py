"""
plots/network.py — Protocol comparison, latency breakdown, fairness, profile breakdown
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import SIM_DURATION_S, PROTOCOLS, TRAFFIC_PROFILES


# =============================================================================
# Colour palette (consistent with rest of project)
# =============================================================================

PROTO_COLORS = {
    "UDP":         "#e377c2",
    "TCP NewReno": "#1f77b4",
    "TCP CUBIC":   "#2ca02c",
    "TCP BBR":     "#ff7f0e",
    "QUIC":        "#9467bd",
    "QUIC (analytical)": "#9467bd",
}

# Fallback for any unlisted label
_FALLBACK_COLORS = ["#9467bd", "#8c564b", "#17becf", "#bcbd22"]


def _proto_color(label: str, idx: int) -> str:
    return PROTO_COLORS.get(label, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


# =============================================================================
# Protocol comparison bar chart
# =============================================================================

def draw_protocol_comparison(ns3_results: list,
                              out: str = "output/ntn_protocol_comparison.png") -> str:
    """
    Grouped bar chart comparing transport protocols on four metrics:
      - Mean end-to-end latency [ms]
      - Throughput [kbps]
      - Packet loss rate [%]
      - Jitter [ms]

    Parameters
    ----------
    ns3_results : list[dict]
        Output of ntn_ns3.run_ns3_protocol_suite() — one dict per protocol.
        Each dict must contain keys:
          label, mean_delay_ms, throughput_kbps, loss_pct, jitter_ms,
          handovers, elevation_deg, svc_delay_ms.
    out : str
        Output filename.

    Returns
    -------
    str  Path of the saved figure.
    """
    if not ns3_results:
        print("[ProtocolChart]  No results to plot — skipping.")
        return out

    labels    = [r["label"]              for r in ns3_results]
    latencies = [r["mean_delay_ms"]      for r in ns3_results]
    tputs     = [r["throughput_kbps"]    for r in ns3_results]
    losses    = [r["loss_pct"]           for r in ns3_results]
    jitters   = [r.get("jitter_ms", 0.0) for r in ns3_results]
    handovers = [r.get("handovers", 0)   for r in ns3_results]

    colors = [_proto_color(lbl, i) for i, lbl in enumerate(labels)]
    x = np.arange(len(labels))
    bar_w = 0.62

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
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
    ax.set_xticklabels(labels, fontsize=9, rotation=12, ha="right")
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
    ax.set_xticklabels(labels, fontsize=9, rotation=12, ha="right")
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
    ax.set_xticklabels(labels, fontsize=9, rotation=12, ha="right")
    ax.set_ylabel("Packet Loss [%]")
    ax.set_title("Packet Loss")
    ax.grid(axis="y", alpha=0.35)
    loss_top = max(losses) * 1.35 + 0.1
    ax.set_ylim(0, loss_top)
    # Annotate handover counts below each bar
    for xi, ho in zip(x, handovers):
        ax.text(xi, -0.04 * loss_top,
                f"{ho} H/O", ha="center", va="top",
                fontsize=7, color="#555")

    # ── Panel 4: Jitter ───────────────────────────────────────────────────────
    ax = axes[3]
    bars = ax.bar(x, jitters, bar_w, color=colors, edgecolor="#333",
                  linewidth=0.8, alpha=0.88)
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=12, ha="right")
    ax.set_ylabel("Jitter [ms]")
    ax.set_title("Jitter")
    ax.grid(axis="y", alpha=0.35)
    ax.set_ylim(0, max(jitters) * 1.35 + 0.1)

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
# Latency breakdown
# =============================================================================

def draw_latency_breakdown(ns3_results: list,
                            out: str = "output/ntn_latency_breakdown.png") -> str:
    """
    Stacked horizontal bar chart showing per-hop latency breakdown for
    each protocol.

    Hop breakdown:
      - NTN propagation (slant range delay, one-way)
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
    from sim.ns3 import _one_way_delay_ms

    if not ns3_results:
        print("[LatBreakdown]  No results — skipping.")
        return out

    # Fixed hop delays (one-way, ms)
    ntn_delay = _one_way_delay_ms(SAT_HEIGHT_M, 60.0)  # typical 60° elevation

    rows = []
    for i, r in enumerate(ns3_results):
        lbl = r["label"]
        overhead = max(0.0, r["mean_delay_ms"] - ntn_delay)
        rows.append({
            "label":    lbl,
            "color":    _proto_color(lbl, i),
            "ntn":      ntn_delay,
            "overhead": overhead,
        })

    n      = len(rows)
    y      = np.arange(n)
    height = 0.55

    HOP_COLORS = {
        "ntn":      "#42A5F5",
        "overhead": "#EF5350",
    }
    HOP_LABELS = {
        "ntn":      "NTN propagation",
        "overhead": "Protocol overhead",
    }

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.7 + 1.5)))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    hops = ["ntn", "overhead"]
    for hop in hops:
        lefts = np.zeros(n)
        for h in hops:
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
        total = r["ntn"] + r["overhead"]
        ax.text(total + 0.3, i, f"{total:.1f} ms",
                va="center", fontsize=7.5, color="#333")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[LatBreakdown]  Saved -> {out}")
    plt.close()
    return out


# =============================================================================
# Jain's fairness index
# =============================================================================

def draw_fairness(ns3_results: list,
                  out: str = "output/ntn_fairness.png") -> str:
    """
    Bar chart of Jain's fairness index per protocol, with reference lines
    at 0.5 (poor), 0.75 (fair), and 1.0 (perfect).

    Jain's fairness index J = (Σxᵢ)² / (n · Σxᵢ²) where xᵢ is the
    per-flow throughput.  J = 1.0 means perfectly equal sharing;
    J → 1/n means only one flow is active.

    Parameters
    ----------
    ns3_results : list[dict]
        Each dict must contain a ``fairness_index`` key (float in [0, 1]).
    out : str
        Output filename.

    Returns
    -------
    str  Path of the saved figure.
    """
    if not ns3_results:
        print("[Fairness]  No results — skipping.")
        return out

    labels   = [r["label"]                    for r in ns3_results]
    fairness = [r.get("fairness_index", 0.0)  for r in ns3_results]
    colors   = [_proto_color(lbl, i) for i, lbl in enumerate(labels)]

    x     = np.arange(len(labels))
    bar_w = 0.55

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        "Jain's Fairness Index — 5G-NTN Satellite Link\n"
        "J = 1.0 = perfect fairness  |  J → 1/n = one flow dominates",
        fontsize=10,
    )

    bars = ax.bar(x, fairness, bar_w, color=colors, edgecolor="#333",
                  linewidth=0.8, alpha=0.88)

    # Annotate bars with exact value
    for bar, val in zip(bars, fairness):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    # Reference lines
    ref_lines = [(1.00, "#155724", "1.00 — Perfect"),
                 (0.75, "#856404", "0.75 — Fair"),
                 (0.50, "#d62728", "0.50 — Poor")]
    for val, col, lbl in ref_lines:
        ax.axhline(val, color=col, linewidth=1.2, linestyle="--",
                   alpha=0.75, label=lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Jain's Fairness Index", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[Fairness]  Saved -> {out}")
    plt.close()
    return out


# =============================================================================
# Per-traffic-profile breakdown
# =============================================================================

def draw_profile_breakdown(ns3_results: list,
                            out: str = "output/ntn_profile_breakdown.png") -> str:
    """
    Grouped bar chart showing throughput and packet loss rate broken down
    by traffic profile for each protocol.

    Parameters
    ----------
    ns3_results : list[dict]
        Each dict must contain a ``profile_stats`` key mapping profile name
        → {tx, rx, rx_bytes, delay_sum}.
    out : str
        Output filename.

    Returns
    -------
    str  Path of the saved figure.
    """
    from config import SIM_DURATION_S as _SIM_S, TRAFFIC_PROFILES

    if not ns3_results:
        print("[ProfileBreakdown]  No results — skipping.")
        return out

    # Check that profile_stats is actually populated
    has_data = any(
        any(v["rx"] > 0 for v in r.get("profile_stats", {}).values())
        for r in ns3_results
    )
    if not has_data:
        print("[ProfileBreakdown]  profile_stats empty — skipping.")
        return out

    profiles    = list(TRAFFIC_PROFILES.keys())
    proto_labels = [r["label"] for r in ns3_results]
    n_protos     = len(proto_labels)
    n_profiles   = len(profiles)

    base_palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#17becf"]
    PROFILE_COLORS = {p: base_palette[i % len(base_palette)] for i, p in enumerate(profiles)}

    active_s = _SIM_S - 1.0

    # Build per-protocol, per-profile throughput and loss arrays
    tput_matrix = np.zeros((n_protos, n_profiles))
    loss_matrix = np.zeros((n_protos, n_profiles))

    for pi, r in enumerate(ns3_results):
        ps = r.get("profile_stats", {})
        for qi, prof in enumerate(profiles):
            stats = ps.get(prof, {"tx": 0, "rx": 0, "rx_bytes": 0})
            rx_b  = stats.get("rx_bytes", 0)
            tx_n  = max(stats.get("tx", 0), 1)
            rx_n  = stats.get("rx", 0)
            tput_matrix[pi, qi] = rx_b * 8.0 / active_s / 1e3   # kbps
            loss_matrix[pi, qi] = 100.0 * (1 - rx_n / tx_n) if tx_n > 0 else 0.0

    x     = np.arange(n_protos)
    bar_w = 0.8 / n_profiles

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    profile_desc = {
        "streaming": "streaming (2.5 Mbps UDP)",
        "gaming": "gaming (120 kbps UDP)",
        "texting": "texting (bursty 30 kbps UDP)",
        "voice": "voice (32 kbps UDP)",
        "bulk": "bulk (TCP BulkSend)",
    }
    subtitle = "  |  ".join(profile_desc.get(p, p) for p in profiles)
    fig.suptitle(
        "Per-Traffic-Profile Performance Breakdown — 5G-NTN Satellite Link\n"
        f"Profiles: {subtitle}",
        fontsize=10,
    )

    # ── Top: Throughput ───────────────────────────────────────────────────────
    for qi, prof in enumerate(profiles):
        offset = (qi - n_profiles / 2 + 0.5) * bar_w
        vals   = tput_matrix[:, qi]
        bars   = ax1.bar(x + offset, vals, bar_w,
                         color=PROFILE_COLORS[prof],
                         edgecolor="#333", linewidth=0.6, alpha=0.85,
                         label=prof.capitalize())
        ax1.bar_label(bars, fmt="%.0f", fontsize=6.5, padding=1)

    ax1.set_xticks(x)
    ax1.set_xticklabels(proto_labels, fontsize=9)
    ax1.set_ylabel("Throughput [kbps]", fontsize=9)
    ax1.set_title("Throughput by Traffic Profile", fontsize=9)
    ax1.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, tput_matrix.max() * 1.3 + 1)

    # ── Bottom: Packet loss ───────────────────────────────────────────────────
    for qi, prof in enumerate(profiles):
        offset = (qi - n_profiles / 2 + 0.5) * bar_w
        vals   = loss_matrix[:, qi]
        bars   = ax2.bar(x + offset, vals, bar_w,
                         color=PROFILE_COLORS[prof],
                         edgecolor="#333", linewidth=0.6, alpha=0.85,
                         label=prof.capitalize())
        ax2.bar_label(bars, fmt="%.1f%%", fontsize=6.5, padding=1)

    ax2.set_xticks(x)
    ax2.set_xticklabels(proto_labels, fontsize=9)
    ax2.set_ylabel("Packet Loss [%]", fontsize=9)
    ax2.set_title("Packet Loss by Traffic Profile", fontsize=9)
    ax2.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, loss_matrix.max() * 1.35 + 0.5)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[ProfileBreakdown]  Saved -> {out}")
    plt.close()
    return out


# =============================================================================
# Congestion-window dynamics (RFC 5681 recovery trace)
# =============================================================================

def draw_cwnd_dynamics(ns3_results: list,
                        out: str = "output/ntn_cwnd_dynamics.png") -> str:
    """
    Per-protocol effective in-flight window trace derived from the per-second
    throughput time-series via Little's law (cwnd_eff = throughput × RTT).

    Visualises congestion-window recovery after each handover blackout,
    providing a model-independent view of RFC 5681 / BBR behaviour without
    hooking into NS-3 internals.
    """
    if not ns3_results:
        print("[Cwnd]  No results — skipping.")
        return out

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    any_trace = False
    for idx, r in enumerate(ns3_results):
        ts = r.get("timeseries", {})
        t_s = ts.get("t_s", [])
        cwnd_bytes = ts.get("cwnd_eff_bytes", [])
        if not t_s or not cwnd_bytes:
            continue
        any_trace = True
        label = r["label"]
        color = PROTO_COLORS.get(label, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])
        cwnd_kb = [b / 1024.0 for b in cwnd_bytes]
        ax.plot(t_s, cwnd_kb, color=color, lw=1.6, label=label, alpha=0.9)

        # Overlay handover blackouts as shaded regions
        for (t0, t1) in ts.get("handover_times", []):
            ax.axvspan(t0, t1, color="#666", alpha=0.08)

    if not any_trace:
        print("[Cwnd]  timeseries missing cwnd_eff_bytes — skipping.")
        plt.close()
        return out

    ax.set_xlabel("Simulation time [s]", fontsize=10)
    ax.set_ylabel("Effective in-flight window [kB]  (throughput × RTT)", fontsize=10)
    ax.set_title(
        "Congestion-Window Dynamics (RFC 5681 proxy) — shaded = handover blackout",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[Cwnd]  Saved -> {out}")
    plt.close()
    return out
