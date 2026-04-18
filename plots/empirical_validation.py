"""
plots/empirical_validation.py — Four-panel overlay of simulated KPIs vs
published real-world NTN measurements.

Unlike plots/summary.py, which compares the simulation to standards-body
*targets* (3GPP / ITU-R thresholds), this figure overlays every measured
metric against a *published measurement campaign* (Sander IMC 2022,
Michel IMC 2022, 3GPP TR 38.821 Annex B calibration, ITU-R P.618).

Panels
------
  1. Latency vs Sander RTT CDF
     — per-protocol mean latency × 2 (round-trip approx) vs the Sander
       3-point CDF (median 48, p95 65, p99 95 ms).
  2. Loss distribution vs Sander TCP loss CDF
     — per-protocol loss_pct vs Sander median 0.5 % / p95 2.1 % band.
  3. Post-handover throughput recovery
     — timeseries slice from `timeseries['throughput_kbps']` starting
       ~5 s before each handover event, ensemble-averaged and normalised
       by pre-HO mean, vs Michel's 1.2 s recovery-to-90 % reference.
  4. Handover blackout duration vs Sander HO-gap CDF
     — per-slot `interruption_ms` histogram vs Sander median 130 /
       p95 250 ms band.

Use
---
Called once from main.py after `draw_validation_summary`.  Reads the
same `direct_results` list plus `config.EMPIRICAL_REFS` — no new probe
is required in sim/ns3.py.
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import EMPIRICAL_REFS
from plots.network import _proto_color


def _fmt_caption(entry: dict) -> str:
    """Return a short citation tag like 'Sander2022 Fig.5'."""
    src = entry.get("source", "")
    # Keep first ~32 chars of the source string
    return src.split("—")[0].strip()[:40]


def _post_ho_recovery_curve(ns3_results: list,
                            window_s: float = 8.0) -> tuple:
    """
    Build an ensemble-averaged post-handover throughput curve.

    Parameters
    ----------
    ns3_results : list of per-protocol result dicts.  Each must contain
        'timeseries' with keys t_s, throughput_kbps, handover_times.
    window_s : total window length (pre + post HO) in seconds.

    Returns
    -------
    (t_rel, norm_tput)
        t_rel: 1-D array of relative time [s], 0 = HO event.
        norm_tput: 1-D array of ensemble-mean throughput normalised by
        the pre-HO mean (1.0 = steady-state).  NaN when no HO events.
    """
    pre_s  = 2.0
    post_s = window_s - pre_s

    segments = []
    for r in ns3_results:
        ts = r.get("timeseries") or {}
        t_arr = np.asarray(ts.get("t_s", []), dtype=float)
        y_arr = np.asarray(ts.get("throughput_kbps", []), dtype=float)
        ho_events = ts.get("handover_times", [])
        if t_arr.size == 0 or not ho_events:
            continue
        for (t_blk_start, _t_blk_end) in ho_events:
            mask = (t_arr >= t_blk_start - pre_s) & (t_arr <= t_blk_start + post_s)
            if mask.sum() < 3:
                continue
            t_seg = t_arr[mask] - t_blk_start    # 0 = HO event
            y_seg = y_arr[mask]
            pre_mean = y_arr[(t_arr >= t_blk_start - pre_s) &
                              (t_arr <  t_blk_start)].mean()
            if not np.isfinite(pre_mean) or pre_mean <= 0:
                continue
            segments.append((t_seg, y_seg / pre_mean))

    if not segments:
        return np.linspace(-pre_s, post_s, 20), np.full(20, np.nan)

    grid = np.linspace(-pre_s, post_s, 40)
    stack = np.array([np.interp(grid, t, y, left=np.nan, right=np.nan)
                      for (t, y) in segments])
    return grid, np.nanmean(stack, axis=0)


def draw_empirical_validation(ns3_results: list,
                               channel_stats: list = None,
                               out: str = "output/ntn_empirical_validation.png"
                               ) -> str:
    """
    Render the four-panel empirical-overlay validation figure.

    Returns the output path on success, or ``out`` unchanged on early exit
    (empty ns3_results).
    """
    if not ns3_results:
        print("[EmpiricalVal]  No NS-3 results — skipping.")
        return out

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor("#f8f9fa")
    for ax in axes.flat:
        ax.set_facecolor("#f8f9fa")

    ref_lat  = EMPIRICAL_REFS["latency"]
    ref_loss = EMPIRICAL_REFS["loss"]
    ref_ho   = EMPIRICAL_REFS["handover"]

    labels = [r["label"] for r in ns3_results]
    colors = [_proto_color(lbl, i) for i, lbl in enumerate(labels)]
    x_pos  = np.arange(len(labels))

    # ── Panel 1: Latency overlay ──────────────────────────────────────────────
    ax1 = axes[0, 0]
    # Simulation mean latency is one-way service→server; double it to
    # approximate the round-trip quantity Sander measures.
    rtt_est_ms = np.array([2.0 * r.get("mean_delay_ms", 0.0)
                           for r in ns3_results])
    bars1 = ax1.bar(x_pos, rtt_est_ms, color=colors, edgecolor="white",
                    linewidth=0.5, alpha=0.9, label="Simulated 2× one-way delay")

    for key, style, lbl in [
        ("rtt_median_ms", dict(color="#1b5e20", linestyle="-",  linewidth=1.8),
                                                                "Sander median (48 ms)"),
        ("rtt_p95_ms",    dict(color="#e65100", linestyle="--", linewidth=1.4),
                                                                "Sander p95 (65 ms)"),
        ("rtt_p99_ms",    dict(color="#b71c1c", linestyle=":",  linewidth=1.4),
                                                                "Sander p99 (95 ms)"),
    ]:
        v = ref_lat[key]["value"]
        ax1.axhline(v, label=lbl, **style)

    for bar, v in zip(bars1, rtt_est_ms):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.0f}",
                 ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("Round-trip latency [ms]", fontsize=10)
    ax1.set_title("RTT vs Sander IMC 2022 Starlink measurements",
                  fontsize=11, weight="bold")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, max(120, float(rtt_est_ms.max()) * 1.15))

    # ── Panel 2: Loss overlay ─────────────────────────────────────────────────
    ax2 = axes[0, 1]
    loss_arr = np.array([r.get("loss_pct", 0.0) for r in ns3_results])
    bars2 = ax2.bar(x_pos, loss_arr, color=colors, edgecolor="white",
                    linewidth=0.5, alpha=0.9, label="Simulated loss")

    med_loss = ref_loss["median_pct"]["value"]
    p95_loss = ref_loss["p95_pct"]["value"]
    ax2.axhspan(med_loss, p95_loss, color="#fff3e0", alpha=0.6,
                label=f"Sander measured band ({med_loss:.1f}–{p95_loss:.1f}%)")
    ax2.axhline(med_loss, color="#1b5e20", linewidth=1.6,
                label=f"Sander median ({med_loss:.1f}%)")
    ax2.axhline(p95_loss, color="#e65100", linewidth=1.3, linestyle="--",
                label=f"Sander p95 ({p95_loss:.1f}%)")

    for bar, v in zip(bars2, loss_arr):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 v + 0.05 * max(p95_loss, v + 1e-3),
                 f"{v:.2f}%",
                 ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Packet loss [%]", fontsize=10)
    ax2.set_title("Loss rate vs Sander IMC 2022 TCP flow CDF",
                  fontsize=11, weight="bold")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, max(3.5, float(loss_arr.max()) * 1.25))

    # ── Panel 3: Post-handover throughput recovery ────────────────────────────
    ax3 = axes[1, 0]
    t_rel, norm = _post_ho_recovery_curve(ns3_results, window_s=8.0)

    if np.isnan(norm).all():
        ax3.text(0.5, 0.5,
                 "No handover events captured in this run",
                 transform=ax3.transAxes, ha="center", va="center",
                 fontsize=11, color="#666")
    else:
        ax3.plot(t_rel, norm, color="#1f77b4", linewidth=2.0,
                 label="Simulated (ensemble mean across HOs)")
        ax3.axvspan(0.0, 0.2, color="#fdecea", alpha=0.8,
                    label="HO blackout window")
        # Michel 1.2 s recovery target → 90 % of pre-HO mean at t=1.2 s
        rec_s = ref_ho["recovery_to_90pct_s"]["value"]
        ax3.axvline(rec_s, color="#b71c1c", linewidth=1.5, linestyle="--",
                    label=f"Michel recovery-to-90% ({rec_s:.1f} s)")
        ax3.axhline(0.9, color="#1b5e20", linewidth=1.2, linestyle=":",
                    alpha=0.8, label="90 % of pre-HO mean")

    ax3.set_xlabel("Time relative to HO event [s]", fontsize=10)
    ax3.set_ylabel("Throughput / pre-HO mean", fontsize=10)
    ax3.set_title("Post-handover recovery vs Michel IMC 2022",
                  fontsize=11, weight="bold")
    ax3.legend(fontsize=8, loc="lower right")
    ax3.grid(alpha=0.3)

    # ── Panel 4: Handover blackout duration ───────────────────────────────────
    ax4 = axes[1, 1]
    ho_ms = []
    for r in ns3_results:
        for slot in r.get("schedule", []) or []:
            if slot.get("interruption_ms", 0.0) > 0:
                ho_ms.append(float(slot["interruption_ms"]))
    ho_ms_arr = np.array(ho_ms)

    if ho_ms_arr.size == 0:
        ax4.text(0.5, 0.5,
                 "No handover events captured in this run",
                 transform=ax4.transAxes, ha="center", va="center",
                 fontsize=11, color="#666")
    else:
        bins = np.linspace(0, max(320, float(ho_ms_arr.max()) * 1.1), 15)
        ax4.hist(ho_ms_arr, bins=bins, color="#6a9bcb",
                 edgecolor="white", alpha=0.9,
                 label=f"Simulated ({ho_ms_arr.size} HOs)")

        med_ho = ref_ho["interruption_median_ms"]["value"]
        p95_ho = ref_ho["interruption_p95_ms"]["value"]
        ax4.axvline(med_ho, color="#1b5e20", linewidth=1.8,
                    label=f"Sander median ({med_ho:.0f} ms)")
        ax4.axvline(p95_ho, color="#e65100", linewidth=1.4, linestyle="--",
                    label=f"Sander p95 ({p95_ho:.0f} ms)")

    ax4.set_xlabel("HO interruption duration [ms]", fontsize=10)
    ax4.set_ylabel("Count", fontsize=10)
    ax4.set_title("HO blackout vs Sander IMC 2022 Tab.4",
                  fontsize=11, weight="bold")
    ax4.legend(fontsize=8, loc="upper right")
    ax4.grid(axis="y", alpha=0.3)

    # ── Figure-level title and caption ────────────────────────────────────────
    fig.suptitle("NTN Simulation — Empirical Real-World Validation",
                 fontsize=13, weight="bold", y=0.995)
    fig.text(0.5, 0.01,
             "References:  Sander et al. IMC 2022 Starlink measurements   |   "
             "Michel et al. IMC 2022 Starlink performance   |   "
             "ITU-R P.618 / 3GPP TR 38.821",
             ha="center", fontsize=8, style="italic", color="#555")

    plt.tight_layout(rect=[0, 0.025, 1, 0.97])
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[EmpiricalVal]  Saved -> {out}")
    plt.close(fig)
    return out
