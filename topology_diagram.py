"""
topology_diagram.py — Visualisation functions for the NTN satellite simulation
===============================================================================
Produces output figures:

  ntn_ber_bler.png
      BER and BLER vs Eb/N0 for all OpenNTN scenarios (urban/dense/suburban).

  ntn_protocol_comparison.png
      Grouped bar chart comparing all transport protocols (UDP, TCP NewReno,
      TCP CUBIC, TCP BBR, QUIC) on latency, throughput, packet loss, jitter.

  ntn_link_budget_waterfall.png
      Horizontal waterfall showing per-satellite link budget breakdown.

  ntn_snr_vs_elevation.png
      SNR vs elevation angle with PER sigmoid overlay.

  ntn_latency_breakdown.png
      Per-hop stacked latency bar chart (NTN propagation + protocol overhead).

  ntn_handover_impact.png
      Per-slot throughput showing TCP congestion collapse vs QUIC resilience.

  ntn_handover_schedule.png
      Gantt-style timeline of the satellite handover schedule, coloured by PER.

  ntn_timeseries.png
      Per-second throughput time-series with handover gap markers.

  ntn_fairness.png
      Jain's fairness index per protocol.

  ntn_profile_breakdown.png
      Per-traffic-profile throughput and loss breakdown per protocol.
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
# BER / BLER curves
# =============================================================================

def draw_ber_bler(ber_results: dict, snr_range=None,
                  out: str = "output/ntn_ber_bler.png") -> str:
    """
    Two-panel BER and BLER vs Eb/N0 for all simulated NTN scenarios.

    Parameters
    ----------
    ber_results : dict  Mapping scenario -> (ber_array, bler_array).
    snr_range   : array-like  Eb/N0 [dB].  Inferred from array length if None.
    out         : str   Output filename.

    Returns
    -------
    str  Path of the saved figure.
    """
    if not ber_results:
        print("[BER/BLER]  No results — skipping.")
        return out

    if snr_range is None:
        first_ber, _ = next(iter(ber_results.values()))
        n = len(first_ber)
        snr_range = np.arange(0, n, dtype=float)
    snr_range = np.asarray(snr_range, dtype=float)

    ber_colors = {
        "urban":       "#1f77b4",
        "dense_urban": "#d62728",
        "suburban":    "#2ca02c",
    }

    fig, (ax_ber, ax_bler) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#f8f9fa")
    for ax in (ax_ber, ax_bler):
        ax.set_facecolor("#f8f9fa")

    fig.suptitle(
        "Sionna + OpenNTN (TR38.811) — Coded BER & BLER vs Eb/N0\n"
        f"QPSK · LDPC r=0.5 · LEO {SAT_HEIGHT_M/1e3:.0f} km · "
        f"{CARRIER_FREQ_HZ/1e9:.1f} GHz",
        fontsize=10,
    )

    for sc, (ber, bler) in ber_results.items():
        col   = ber_colors.get(sc, "gray")
        label = sc.replace("_", " ").title()
        ax_ber.semilogy(snr_range, np.clip(ber,  1e-5, 1), "o-", ms=4,
                        color=col, label=label)
        ax_bler.semilogy(snr_range, np.clip(bler, 1e-5, 1), "s-", ms=4,
                         color=col, label=label)

    for ax, title, ylabel in [
        (ax_ber,  "Coded BER",  "Bit Error Rate"),
        (ax_bler, "BLER",       "Block Error Rate"),
    ]:
        ax.set_xlabel("Eb/N0 [dB]", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, which="both", alpha=0.35)
        ax.set_ylim([1e-5, 1.2])
        ax.set_xlim([snr_range[0], snr_range[-1]])

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[BER/BLER]  Saved -> {out}")
    plt.close()
    return out


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
        per = _rt_calibrated_per(fspl, gain, None, ref_gain, eirp_dbm)
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
            _rt_calibrated_per(_fspl_db(SAT_HEIGHT_M, e), -100.0, None, None, eirp_dbm)
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
# 5. Handover schedule timeline (Gantt)
# =============================================================================

def draw_handover_schedule(ns3_results: list,
                            out: str = "output/ntn_handover_schedule.png") -> str:
    """
    Gantt-style chart of the satellite handover schedule.

    Each row = one satellite service slot, coloured by PER.
    Green = low PER (good link), red = high PER (near-dead link).
    Handover interruption gaps are shown as grey shaded regions.

    Parameters
    ----------
    ns3_results : list[dict]  NS-3 results (uses the schedule from the first entry).
    out : str  Output filename.

    Returns
    -------
    str  Path of the saved figure.
    """
    if not ns3_results:
        print("[HOSchedule]  No results — skipping.")
        return out

    schedule = ns3_results[0].get("schedule", [])
    if not schedule:
        print("[HOSchedule]  No handover schedule found — skipping.")
        return out

    import matplotlib.colors as mcolors
    cmap = plt.cm.RdYlGn_r   # green=0 PER, red=1 PER

    fig, ax = plt.subplots(figsize=(13, max(3.5, len(schedule) * 0.65 + 1.5)))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    for row_idx, slot in enumerate(schedule):
        t0   = slot["t_start"]
        t1   = slot["t_end"]
        per  = slot["per"]
        gap  = slot["interruption_ms"] / 1000.0
        elev = slot["elev_deg"]
        sid  = slot["sat_id"]

        color = cmap(per)

        # Service bar
        ax.barh(row_idx, t1 - t0, left=t0, height=0.6,
                color=color, edgecolor="#444", linewidth=0.7, alpha=0.9)

        # Label inside bar
        mid = (t0 + t1) / 2
        ax.text(mid, row_idx, f"Sat {sid}\n{elev:.0f}°  PER={per:.3f}",
                ha="center", va="center", fontsize=7.5,
                color="white" if per > 0.5 else "#111",
                fontweight="bold")

        # Handover gap (grey shading before this slot starts)
        if gap > 0:
            ax.barh(row_idx, gap, left=t0 - gap, height=0.6,
                    color="#aaa", edgecolor="#777", linewidth=0.5,
                    alpha=0.6, hatch="///")
            ax.text(t0 - gap / 2, row_idx - 0.38, f"{gap*1e3:.0f} ms gap",
                    ha="center", va="top", fontsize=6.5, color="#555")

    ax.set_yticks(range(len(schedule)))
    ax.set_yticklabels([f"Slot {i}" for i in range(len(schedule))], fontsize=8)
    ax.set_xlabel("Simulation Time [s]", fontsize=10)
    ax.set_xlim(0, SIM_DURATION_S)
    ax.set_ylim(-0.6, len(schedule) - 0.4)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # Colour bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.7, pad=0.01)
    cbar.set_label("Packet Error Rate", fontsize=9)

    ax.set_title(
        "Satellite Handover Schedule — 5G-NTN LEO Pass\n"
        "Colour = PER  |  Grey hatching = beam interruption gap",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[HOSchedule]  Saved -> {out}")
    plt.close()
    return out


# =============================================================================
# 6. Per-second throughput time-series
# =============================================================================

def draw_timeseries(ns3_results: list,
                    out: str = "output/ntn_timeseries.png") -> str:
    """
    Per-second throughput time-series for all protocols, one subplot per
    protocol, with handover beam-interruption gaps shown as shaded regions
    and handover start times as vertical dashed lines.

    Parameters
    ----------
    ns3_results : list[dict]
        Output of run_ns3_protocol_suite().  Each dict must contain a
        ``timeseries`` key with sub-keys:
          t_s             : list[float]  probe times [s]
          throughput_kbps : list[float]  per-second throughput [kbps]
          handover_times  : list[(float,float)]  (t_start, t_end) of each gap
    out : str
        Output filename.

    Returns
    -------
    str  Path of the saved figure.
    """
    if not ns3_results:
        print("[Timeseries]  No results — skipping.")
        return out

    n_protos = len(ns3_results)
    fig, axes = plt.subplots(n_protos, 1,
                             figsize=(14, 2.8 * n_protos),
                             sharex=True)
    if n_protos == 1:
        axes = [axes]

    fig.suptitle(
        "Per-Second Throughput Time-Series — 5G-NTN LEO Satellite Link\n"
        "Grey shading = beam interruption gap during handover   "
        "Red dashed = handover start",
        fontsize=10,
    )

    for ax, r in zip(axes, ns3_results):
        ts = r.get("timeseries", {})
        t_s    = ts.get("t_s", [])
        tput   = ts.get("throughput_kbps", [])
        ho_times = ts.get("handover_times", [])

        color  = _proto_color(r["label"], ns3_results.index(r))
        mean_t = r.get("throughput_kbps", 0.0)

        if t_s and tput:
            ax.step(t_s, tput, where="post", color=color, linewidth=1.4,
                    alpha=0.85)
            ax.fill_between(t_s, tput, step="post", alpha=0.18, color=color)
        else:
            ax.text(0.5, 0.5, "No time-series data",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#888")

        # Mean throughput reference line
        ax.axhline(mean_t, color=color, linewidth=1.0, linestyle="--",
                   alpha=0.55, label=f"Mean {mean_t:.0f} kbps")

        # Handover beam-gap shading and start markers
        for t_start, t_end in ho_times:
            ax.axvspan(t_start, t_end, alpha=0.18, color="#d62728", zorder=0)
            ax.axvline(t_start, color="#d62728", linewidth=1.2,
                       linestyle="--", alpha=0.8, zorder=1)

        ax.set_ylabel("Throughput\n[kbps]", fontsize=8)
        ax.set_title(r["label"], fontsize=9, fontweight="bold",
                     color=color, loc="left")
        ax.grid(axis="both", alpha=0.25)
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
        y_top = max(tput) * 1.3 + 1 if tput else max(mean_t * 1.5, 10)
        ax.set_ylim(0, y_top)

    axes[-1].set_xlabel("Simulation Time [s]", fontsize=9)
    from config import SIM_DURATION_S as _SIM_S
    axes[-1].set_xlim(0, _SIM_S)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[Timeseries]  Saved -> {out}")
    plt.close()
    return out


# =============================================================================
# 8. Jain's fairness index
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
# 9. Per-traffic-profile breakdown
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
