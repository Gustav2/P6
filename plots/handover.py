"""
plots/handover.py — Handover impact, handover schedule, timeseries
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SIM_DURATION_S
from plots.network import _proto_color


# Slot color constants — used by draw_handover_impact to mark satellites
# consistently.
_SLOT_COLORS = ["#1a6fc4", "#2ca02c", "#d62728"]


# =============================================================================
# 4. Handover impact
# =============================================================================

def draw_handover_impact(ns3_results: list,
                          out: str = "output/ntn_handover_impact.png") -> str:
    """
    Per-slot throughput bar chart showing the impact of satellite handovers
    on each protocol.

    The per-slot throughput is reconstructed from aggregate throughput by
    distributing proportionally to (1-PER) per slot; this is an estimate,
    not a directly measured slot-throughput trace.

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

    slot_labels = []
    for s in schedule:
        lbl = s.get("label") or f"Sat {s['sat_id']}"
        slot_labels.append(f"{lbl}\nPER={s['per']:.3f}")

    bar_w  = 0.75 / n_protos
    x_base = np.arange(n_slots)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Per-Slot Throughput — Handover Impact on Each Protocol\n"
        "Reconstructed estimate from aggregate throughput and slot PER",
        fontsize=10,
    )

    for pi, r in enumerate(ns3_results):
        slot_tput = _per_slot_tput(r["throughput_kbps"], schedule)
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
