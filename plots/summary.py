"""
plots/summary.py — Validation summary table (KPI pass/fail vs standards)
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    SERVICE_LINK_RATE_MBPS,
    HANDOVER_INTERRUPTION_MS_MAX,
)


def _median(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan")
    xs = sorted(xs)
    n = len(xs)
    if n % 2 == 0:
        return 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    return xs[n // 2]


def draw_validation_summary(ns3_results: list,
                             channel_stats: list,
                             ber_results: dict = None,
                             out: str = "output/ntn_validation_summary.png") -> str:
    """
    Render a single-page KPI validation table that cross-references each
    measured metric against a standards-specified target and marks it
    pass/fail.

    Rows are aggregated across protocols by taking the median measured
    value where appropriate (throughput efficiency, latency p95, loss,
    fairness), and the worst-case where appropriate (handover success,
    MOS, rebuffering).

    Standards referenced:
      - 3GPP TR 38.811 §6.7.2 / §7.3  (PHY / BLER)
      - 3GPP TS 38.300 §10.1.2.3       (handover success)
      - ITU-R P.618-13                 (rain fade, scintillation)
      - ITU-T G.107                    (VoIP MOS)
      - RFC 5681                       (TCP cwnd)
      - ETSI TR 103 559                (streaming NTN validation)
    """
    if not ns3_results:
        print("[ValidationSummary]  No NS-3 results — skipping.")
        return out

    # ── Aggregate KPIs across protocols ───────────────────────────────────────
    throughputs_kbps = [r.get("throughput_kbps", 0.0) for r in ns3_results]
    median_tput_mbps = _median(throughputs_kbps) / 1e3
    shannon_ceiling_mbps = SERVICE_LINK_RATE_MBPS
    tput_efficiency_pct = (100.0 * median_tput_mbps / shannon_ceiling_mbps
                           if shannon_ceiling_mbps > 0 else 0.0)

    latencies_ms = [r.get("mean_delay_ms", 0.0) for r in ns3_results
                    if r.get("mean_delay_ms", 0.0) > 0]
    median_latency_ms = _median(latencies_ms)

    losses_pct = [r.get("loss_pct", 0.0) for r in ns3_results]
    median_loss_pct = _median(losses_pct)

    fairness_vals = [r.get("fairness_index", 0.0) for r in ns3_results]
    median_fairness = _median(fairness_vals)

    ho_success_rates = [r.get("handover_success_rate", 1.0) for r in ns3_results]
    worst_ho_success = min(ho_success_rates) if ho_success_rates else 1.0

    rebuffer_pcts = [r.get("rebuffer_ratio_pct", 0.0) for r in ns3_results]
    worst_rebuffer = max(rebuffer_pcts) if rebuffer_pcts else 0.0

    psnrs = [r.get("stream_psnr_db", 0.0) for r in ns3_results
             if r.get("stream_psnr_db", 0.0) > 0]
    worst_psnr = min(psnrs) if psnrs else 0.0

    gaming_latencies = [r.get("gaming_latency_ms") for r in ns3_results
                        if r.get("gaming_latency_ms") is not None]
    worst_gaming_latency = (max(gaming_latencies) if gaming_latencies
                            else float("nan"))

    mos_vals = [r.get("voice_mos", 0.0) for r in ns3_results]
    worst_mos = min(mos_vals) if mos_vals else 0.0

    overhead_vals = [r.get("protocol_overhead_pct", 0.0) for r in ns3_results]
    max_overhead = max(overhead_vals) if overhead_vals else 0.0

    # Cross-layer R²: recompute from channel_stats + schedule
    r_k_per = float("nan")
    if channel_stats and ns3_results[0].get("schedule"):
        sat_to_k = {s["sat_id"]: s.get("k_factor_db", float("nan"))
                    for s in channel_stats}
        pairs = [(sat_to_k.get(slot["sat_id"], float("nan")),
                  slot.get("per", 0.0))
                 for slot in ns3_results[0]["schedule"]]
        pairs = [(k, p) for (k, p) in pairs if not math.isnan(k)]
        if len(pairs) >= 2:
            k_arr = np.array([p[0] for p in pairs])
            per_arr = np.array([p[1] for p in pairs])
            if np.std(k_arr) > 0 and np.std(per_arr) > 0:
                r_k_per = float(np.corrcoef(k_arr, per_arr)[0, 1])

    # BLER deviation (optional — only if ber_results available)
    bler_dev_db = float("nan")
    if ber_results:
        try:
            snr_range = np.arange(0.0, float(len(next(iter(ber_results.values()))[1])), 1.0)
            snr_lin = 10.0 ** (snr_range / 10.0)
            ber_unc = 0.5 * np.array([math.erfc(math.sqrt(max(x, 0.0)))
                                       for x in snr_lin])
            gpp_bler = np.clip(1.0 - (1.0 - ber_unc) ** 864, 1e-5, 1.0)
            urban_bler = np.asarray(
                ber_results.get("urban", next(iter(ber_results.values())))[1],
                dtype=float,
            )
            band = (urban_bler >= 1e-3) & (urban_bler <= 1e-1)
            if band.sum() >= 2:
                ref_snr_at = np.interp(urban_bler[band],
                                        gpp_bler[::-1], snr_range[::-1])
                bler_dev_db = float(np.mean(np.abs(snr_range[band] - ref_snr_at)))
        except Exception:
            pass

    # ── Row definitions: (name, measured_str, target_str, source, pass?) ───
    def _fmt(v, unit="", prec=2):
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return "N/A"
        return f"{v:.{prec}f}{unit}"

    rows = [
        ("Throughput efficiency",
         f"{tput_efficiency_pct:.0f}% of {shannon_ceiling_mbps:.0f} Mbps",
         "> 90% of beam capacity",
         "3GPP TR 38.821 §6.1.1",
         tput_efficiency_pct > 90.0),

        ("Mean latency",
         _fmt(median_latency_ms, " ms", 1),
         "< 100 ms (LEO p95 budget)",
         "ITU-R P.618 / Sander et al. IMC 2022",
         (not math.isnan(median_latency_ms)) and median_latency_ms < 100.0),

        ("Packet loss",
         _fmt(median_loss_pct, " %", 2),
         "< 3% (VoIP class)",
         "ITU-T G.107 / RFC 3393",
         (not math.isnan(median_loss_pct)) and median_loss_pct < 3.0),

        ("Jain's fairness",
         _fmt(median_fairness, "", 3),
         "> 0.9",
         "Jain, Chiu & Hawe (1984)",
         (not math.isnan(median_fairness)) and median_fairness > 0.9),

        ("Handover success rate",
         _fmt(worst_ho_success * 100.0, " %", 1),
         f"> 95% (within {HANDOVER_INTERRUPTION_MS_MAX:.0f} ms budget)",
         "3GPP TS 38.300 §10.1.2.3",
         worst_ho_success >= 0.95),

        ("Streaming rebuffering",
         _fmt(worst_rebuffer, " %", 2),
         "< 5%",
         "ETSI TR 103 559 §5",
         worst_rebuffer < 5.0),

        ("Video PSNR (per-client)",
         _fmt(worst_psnr, " dB", 2),
         "> 30 dB",
         "Empirical H.264 CBR map",
         worst_psnr > 30.0),

        ("Gaming E2E latency",
         _fmt(worst_gaming_latency, " ms", 1),
         "< 50 ms",
         "3GPP TR 22.261 §7.1 (URLLC)",
         (not math.isnan(worst_gaming_latency)) and worst_gaming_latency < 50.0),

        ("VoIP MOS",
         _fmt(worst_mos, "", 3),
         "> 3.5",
         "ITU-T G.107 E-model",
         worst_mos > 3.5),

        ("Protocol overhead",
         _fmt(max_overhead, " %", 2),
         "< 10%",
         "IETF RFC 791 / 9293 header sizes",
         max_overhead < 10.0),

        ("Cross-layer correlation R",
         _fmt(r_k_per, "", 2),
         "|R| > 0.8 (K-factor ↔ PER)",
         "Sander et al. IMC 2022 cross-layer",
         (not math.isnan(r_k_per)) and abs(r_k_per) > 0.8),

        ("BLER deviation vs TR 38.811",
         _fmt(bler_dev_db, " dB", 2),
         "< 2 dB RMS",
         "3GPP TR 38.811 §7.3",
         (not math.isnan(bler_dev_db)) and bler_dev_db < 2.0),
    ]

    # ── Render table figure ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 1.0 + 0.45 * len(rows)))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")
    ax.axis("off")

    col_labels = ["KPI", "Measured", "Target", "Standard", "Status"]
    table_data = []
    cell_colors = []
    for (name, measured, target, source, passed) in rows:
        status = "PASS" if passed else "FAIL"
        table_data.append([name, measured, target, source, status])
        row_bg = "#e8f5e9" if passed else "#fdecea"
        cell_colors.append([row_bg] * 5)

    tbl = ax.table(cellText=table_data,
                   colLabels=col_labels,
                   cellColours=cell_colors,
                   loc="center",
                   cellLoc="left",
                   colWidths=[0.22, 0.17, 0.22, 0.29, 0.10])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)

    # Header styling
    for col_idx in range(len(col_labels)):
        header_cell = tbl[0, col_idx]
        header_cell.set_facecolor("#1f77b4")
        header_cell.set_text_props(color="white", weight="bold")

    # Status column: bold pass/fail
    for row_idx, (*_, passed) in enumerate(rows, start=1):
        status_cell = tbl[row_idx, 4]
        status_cell.set_text_props(
            weight="bold",
            color="#1b5e20" if passed else "#b71c1c",
        )

    pass_count = sum(1 for r in rows if r[-1])
    total      = len(rows)
    header_text = (
        "NTN Simulation Validation Summary — "
        f"{pass_count}/{total} KPIs pass standards targets"
    )
    fig.suptitle(header_text, fontsize=12, weight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[ValidationSummary]  Saved -> {out}  ({pass_count}/{total} pass)")
    plt.close()
    return out
