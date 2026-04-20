"""
plots/channel.py — Link budget waterfall, SNR vs elevation, channel validation
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SAT_HEIGHT_M, CARRIER_FREQ_HZ, PROTOCOLS


# 3GPP TR 38.811 Table 6.7.2-3 reference values (Urban NTN, LOS).
# K-factor: mean and ±1σ in dB.
# Delay spread: mean in log10(DS/s) converted to nanoseconds.
_GPP_ELEV       = [10,   20,   30,   45,   60,   75,   90  ]   # degrees
_GPP_K_MEAN_DB  = [ 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]  # dB (mean)
_GPP_K_STD_DB   = [ 3.5,  3.5,  4.0,  4.0,  4.0,  3.5,  3.5]  # dB (σ)
# log10(DS/s) mean → DS in ns = 10^(μ + 9)
_GPP_DS_MU      = [-7.0, -7.2, -7.5, -7.8, -8.0, -8.3, -8.5]  # log10(s)
_GPP_DS_MEAN_NS = [10 ** (m + 9) for m in _GPP_DS_MU]


def _fspl_db(elevation_deg: float,
             freq_hz: float = CARRIER_FREQ_HZ,
             height_m: float = SAT_HEIGHT_M) -> float:
    """Free-space path loss [dB] at real orbital altitude for a given elevation."""
    RE  = 6_371_000.0
    e   = math.radians(max(float(elevation_deg), 0.5))
    d   = math.sqrt((RE + height_m) ** 2 - (RE * math.cos(e)) ** 2) - RE * math.sin(e)
    lam = 3e8 / freq_hz
    return 20.0 * math.log10(4.0 * math.pi * d / lam)


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
    from config import (PHONE_EIRP_DBM,
                        SAT_HEIGHT_M, CARRIER_FREQ_HZ,
                        SAT_RX_ANTENNA_GAIN_DB,
                        NOISE_FLOOR_DBM, SNR_THRESH_DB,
                        ATMO_GASEOUS_DB)
    from sim.ns3 import (_fspl_db as _ns3_fspl_db,
                         _itu_p618_rain_fade_db,
                         _itu_p618_scintillation_db,
                         _rt_calibrated_per)

    import math as _math
    # Sort by elevation descending
    stats = sorted(channel_stats, key=lambda s: s["elevation_deg"], reverse=True)
    # Use normalised gains (proxy-FSPL-corrected) for the reference — consistent
    # with the ns3.py link budget computation.
    valid_norm = [s.get("normalized_gain_db", float("nan")) for s in stats
                  if not _math.isnan(s.get("normalized_gain_db", float("nan")))]
    ref_gain = max(valid_norm) if valid_norm else float("nan")

    n_sats = len(stats)

    # Build budget stages for each sat
    def _budget(stat, eirp_dbm):
        elev_deg   = max(stat["elevation_deg"], 1.0)
        fspl       = _ns3_fspl_db(SAT_HEIGHT_M, elev_deg)
        # Atmospheric loss (ITU-R P.676 gaseous + P.618 rain + scintillation)
        # applied with the same formula as sim/ns3.py:_rt_calibrated_per so
        # the waterfall's final SNR matches the value that actually drives PER.
        atm_loss = (ATMO_GASEOUS_DB
                    + _itu_p618_rain_fade_db(elev_deg)
                    + _itu_p618_scintillation_db(elev_deg))
        norm_gain  = stat.get("normalized_gain_db", float("nan"))
        if _math.isnan(norm_gain):
            urban = -10.0
        elif not _math.isnan(ref_gain):
            urban = float(np.clip(norm_gain - ref_gain, -12.0, 12.0))
        else:
            urban = 0.0
        snr = (eirp_dbm - fspl - atm_loss + urban
               + SAT_RX_ANTENNA_GAIN_DB - NOISE_FLOOR_DBM)
        per = _rt_calibrated_per(fspl, norm_gain, stat.get("normalized_p10_db"),
                                 ref_gain, None, None,
                                 elev_deg=stat["elevation_deg"])
        return dict(eirp=eirp_dbm, fspl=fspl, atm=atm_loss, urban=urban,
                    noise=NOISE_FLOOR_DBM, snr=snr, thresh=SNR_THRESH_DB,
                    margin=snr - SNR_THRESH_DB, per=per)

    budgets = [_budget(s, PHONE_EIRP_DBM) for s in stats]

    fig, axes = plt.subplots(1, n_sats, figsize=(6 * n_sats, 5),
                              sharex=False)
    if n_sats == 1:
        axes = [axes]
    freq_ghz = CARRIER_FREQ_HZ / 1e9
    fig.suptitle(
        "Link Budget Waterfall — Per Satellite\n"
        f"TX EIRP → FSPL → Atm loss → Urban correction → SNR → Margin vs threshold  "
        f"({freq_ghz:.1f} GHz n255, LEO {SAT_HEIGHT_M/1e3:.0f} km, Phone EIRP {PHONE_EIRP_DBM:.0f} dBm)",
        fontsize=10,
    )

    stage_colors = {
        "TX EIRP":        "#4CAF50",
        "−FSPL":          "#F44336",
        "−Atm loss":      "#6D4C41",
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
            ("−Atm loss",   -budget["atm"],               False),
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
    from config import (PHONE_EIRP_DBM, SAT_HEIGHT_M, SAT_RX_ANTENNA_GAIN_DB,
                        NOISE_FLOOR_DBM, SNR_THRESH_DB, SIGMOID_SLOPE)
    from sim.ns3 import _fspl_db as _ns3_fspl_db, _rt_calibrated_per

    # Slot color constants — used by draw_snr_vs_elevation and draw_handover_impact
    # to mark the three simulated satellites consistently.
    _SLOT_COLORS = ["#1a6fc4", "#2ca02c", "#d62728"]

    elev = np.linspace(1.0, 90.0, 300)

    def _snr_curve(eirp_dbm):
        fspl = np.array([_ns3_fspl_db(SAT_HEIGHT_M, e) for e in elev])
        return eirp_dbm - fspl + SAT_RX_ANTENNA_GAIN_DB - NOISE_FLOOR_DBM

    # Reference normalized gain baseline for PER plotting
    valid_norm = [s.get("normalized_gain_db", float("nan")) for s in channel_stats
                  if not math.isnan(s.get("normalized_gain_db", float("nan")))]
    ref_gain = max(valid_norm) if valid_norm else None

    def _per_curve(eirp_dbm):
        return np.array([
            _rt_calibrated_per(
                _ns3_fspl_db(SAT_HEIGHT_M, e),
                rt_mean_gain_db=(ref_gain if ref_gain is not None else float("nan")),
                rt_gain_p10_db=None,
                rt_ref_gain_db=ref_gain,
                snr_thresh_db=SNR_THRESH_DB,
                sigmoid_slope=SIGMOID_SLOPE,
                elev_deg=e,
            )
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

    freq_ghz = CARRIER_FREQ_HZ / 1e9
    ax1.set_title(
        f"SNR vs Elevation Angle — Phone EIRP ({PHONE_EIRP_DBM:.0f} dBm)\n"
        f"LEO {SAT_HEIGHT_M/1e3:.0f} km  |  {freq_ghz:.1f} GHz n255  |  "
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
# 3. Channel validation
# =============================================================================

def draw_channel_validation(channel_stats: list,
                            out: str = "output/ntn_channel_validation.png") -> str:
    """Three-panel channel validation figure.

    Panel 1 — Free-space path loss vs elevation
        Analytical FSPL curve (solid) and 3GPP TR 38.811 model (FSPL +
        atmospheric absorption + ±1σ shadow-fading band).  Our 8 satellite
        slant-range FSPL values are marked as scatter points.

    Panel 2 — Rician K-factor vs elevation
        RT-computed K-factor (dots) versus 3GPP TR 38.811 Table 6.7.2-3
        reference (mean ± 1σ band).

    Panel 3 — RMS delay spread vs elevation
        RT-computed delay spread (dots) versus 3GPP TR 38.811 Table 6.7.2-3
        reference (mean ± 1σ band).
    """
    # ── Reference curves ──────────────────────────────────────────────────────
    elev_fine = np.linspace(5, 90, 200)
    fspl_fine = np.array([_fspl_db(e) for e in elev_fine])

    # Atmospheric absorption at 2 GHz (clear sky, ITU-R P.676): ~0.3 dB
    # (Less than at 3.5 GHz because O₂/H₂O absorption peaks are above 22 GHz)
    atm_db = 0.3
    # Shadow-fading σ for Urban NTN LOS from TR 38.811 Table 6.7.2-1: 4 dB
    sf_sigma_db = 4.0

    # Interpolated 3GPP reference arrays on fine elevation grid
    gpp_k_mean  = np.interp(elev_fine, _GPP_ELEV, _GPP_K_MEAN_DB)
    gpp_k_upper = gpp_k_mean + np.interp(elev_fine, _GPP_ELEV, _GPP_K_STD_DB)
    gpp_k_lower = gpp_k_mean - np.interp(elev_fine, _GPP_ELEV, _GPP_K_STD_DB)

    gpp_ds_mean  = np.interp(elev_fine, _GPP_ELEV, _GPP_DS_MEAN_NS)
    gpp_ds_upper = gpp_ds_mean * (10 ** 0.4)   # +0.4 in log10
    gpp_ds_lower = gpp_ds_mean / (10 ** 0.4)   # -0.4 in log10

    # ── Per-satellite data from RT ─────────────────────────────────────────────
    valid    = [s for s in channel_stats if s.get("num_paths", 0) > 0]
    rt_elevs = np.array([s["elevation_deg"]                   for s in valid])
    rt_k     = np.array([s.get("k_factor_db", float("nan"))   for s in valid])
    rt_ds    = np.array([s["delay_spread_ns"]                  for s in valid])
    rt_fspl  = np.array([_fspl_db(e)                           for e in rt_elevs])

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    freq_ghz = CARRIER_FREQ_HZ / 1e9
    fig.suptitle(
        f"Channel Validation — 5G-NTN LEO 550 km  |  {freq_ghz:.1f} GHz n255  |  Urban (Munich)\n"
        "RT = single Sionna RT realization (Munich scene, 300 m proxy)    "
        "3GPP = TR 38.811 Table 6.7.2-3 statistical reference (mean ±1σ)",
        fontsize=10,
    )

    # ── Panel 1: Free-space path loss ─────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(elev_fine, fspl_fine,
             color="#1f77b4", lw=2, label="FSPL (analytical)")
    ax1.plot(elev_fine, fspl_fine + atm_db,
             color="#ff7f0e", lw=1.5, ls="--",
             label=f"3GPP TR 38.811\n(FSPL + {atm_db} dB atm.)")
    ax1.fill_between(elev_fine,
                     fspl_fine + atm_db - sf_sigma_db,
                     fspl_fine + atm_db + sf_sigma_db,
                     color="#ff7f0e", alpha=0.15,
                     label=f"±1σ shadow fading ({sf_sigma_db} dB)")
    ax1.scatter(rt_elevs, rt_fspl,
                color="#d62728", zorder=5, s=60,
                label="Our satellites (slant range)")
    for e, f in zip(rt_elevs, rt_fspl):
        ax1.annotate(f"{e:.0f}°", (e, f),
                     textcoords="offset points", xytext=(4, 3),
                     fontsize=7, color="#d62728")

    ax1.set_xlabel("Elevation angle [°]", fontsize=9)
    ax1.set_ylabel("Path loss [dB]", fontsize=9)
    ax1.set_title("Free-Space Path Loss\nvs Elevation", fontsize=9)
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.invert_yaxis()   # higher loss = worse, shown at top
    ax1.grid(alpha=0.3)

    # ── Panel 2: Rician K-factor ──────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(elev_fine, gpp_k_mean,
             color="#2ca02c", lw=2, label="3GPP TR 38.811 (mean)")
    ax2.fill_between(elev_fine, gpp_k_lower, gpp_k_upper,
                     color="#2ca02c", alpha=0.15, label="±1σ (3GPP)")

    k_mask = ~np.isnan(rt_k)
    if k_mask.any():
        ax2.scatter(rt_elevs[k_mask], rt_k[k_mask],
                    color="#d62728", zorder=5, s=60,
                    label="RT — LoS samples (this sim.)")
        for e, k in zip(rt_elevs[k_mask], rt_k[k_mask]):
            ax2.annotate(f"{e:.0f}°", (e, k),
                         textcoords="offset points", xytext=(4, 3),
                         fontsize=7, color="#d62728")
    # Annotate NLoS satellites on the x-axis at y=0 so they are not lost.
    nlos_elevs = rt_elevs[~k_mask]
    if len(nlos_elevs) > 0:
        ax2.scatter(nlos_elevs, np.zeros_like(nlos_elevs),
                    marker="x", color="#666", s=50, zorder=4,
                    label="RT — NLoS (K undefined)")

    ax2.set_xlabel("Elevation angle [°]", fontsize=9)
    ax2.set_ylabel("Rician K-factor [dB]", fontsize=9)
    ax2.set_title("Rician K-Factor\nvs Elevation (Urban NTN LOS)", fontsize=9)
    # Clip to a sensible display range so outliers don't stretch the axis.
    ax2.set_ylim(-5, 25)
    ax2.legend(fontsize=7.5, loc="lower right")
    ax2.grid(alpha=0.3)

    # ── Panel 3: RMS delay spread ──────────────────────────────────────────────
    ax3 = axes[2]
    ax3.semilogy(elev_fine, gpp_ds_mean,
                 color="#9467bd", lw=2, label="3GPP TR 38.811 (mean)")
    ax3.fill_between(elev_fine, gpp_ds_lower, gpp_ds_upper,
                     color="#9467bd", alpha=0.15, label="±1σ (3GPP)")

    ds_mask = rt_ds > 0
    if ds_mask.any():
        ax3.scatter(rt_elevs[ds_mask], rt_ds[ds_mask],
                    color="#d62728", zorder=5, s=60,
                    label="RT computed (this sim.)")
        for e, d in zip(rt_elevs[ds_mask], rt_ds[ds_mask]):
            ax3.annotate(f"{e:.0f}°", (e, d),
                         textcoords="offset points", xytext=(4, 3),
                         fontsize=7, color="#d62728")

    ax3.set_xlabel("Elevation angle [°]", fontsize=9)
    ax3.set_ylabel("RMS delay spread [ns]", fontsize=9)
    ax3.set_title("RMS Delay Spread\nvs Elevation (Urban NTN LOS)", fontsize=9)
    # Clip to a sensible log range so canyon-resonant outliers don't dominate.
    ax3.set_ylim(1, 1e3)
    ax3.legend(fontsize=7.5, loc="upper right")
    ax3.grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[ChannelValidation]  Saved -> {out}")
    plt.close()
    return out


# =============================================================================
# 4. Cross-layer correlation (PHY K-factor ↔ NS-3 loss/latency)
# =============================================================================

def draw_cross_layer_correlation(channel_stats: list, ns3_results: list,
                                  out: str = "output/ntn_cross_layer.png") -> str:
    """
    Per-satellite cross-layer correlation between the PHY-level K-factor and
    the NS-3 service-link PER / slot delay.

    For each satellite in the handover schedule we pair:
      * K-factor [dB] (sampled by the TR 38.811 stochastic channel model)
      * Slot PER [%]  (RT-calibrated packet error rate from the link budget)
      * Slot delay [ms] (one-way propagation delay at the slot elevation)

    Pearson R² is computed between K-factor and both PER and delay.  A strong
    negative correlation (R² > 0.64 ≈ |R| > 0.8) is expected: higher K-factor
    means more direct-path energy and therefore lower PER.
    """
    if not ns3_results:
        print("[CrossLayer]  No NS-3 results — skipping.")
        return out

    # Pull the first result's schedule (all protocols share one schedule)
    schedule = ns3_results[0].get("schedule", [])
    if not schedule:
        print("[CrossLayer]  Empty schedule — skipping.")
        return out

    sat_to_k = {s["sat_id"]: s.get("k_factor_db", float("nan"))
                for s in channel_stats}

    pairs = []
    for slot in schedule:
        sid    = slot["sat_id"]
        k_db   = sat_to_k.get(sid, float("nan"))
        per    = slot.get("per", 0.0)
        delay  = slot.get("delay_ms", 0.0)
        elev   = slot.get("elev_deg", 0.0)
        if not math.isnan(k_db):
            pairs.append((elev, k_db, per, delay, sid))

    if len(pairs) < 2:
        print(f"[CrossLayer]  Not enough LoS samples ({len(pairs)}) — skipping.")
        return out

    elev_arr  = np.array([p[0] for p in pairs])
    k_arr     = np.array([p[1] for p in pairs])
    per_arr   = np.array([p[2] for p in pairs]) * 100.0   # to %
    delay_arr = np.array([p[3] for p in pairs])
    sid_arr   = [p[4] for p in pairs]

    def _pearson_r(x, y):
        if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    r_k_per   = _pearson_r(k_arr, per_arr)
    r_k_delay = _pearson_r(k_arr, delay_arr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#f8f9fa")
    for ax in (ax1, ax2):
        ax.set_facecolor("#f8f9fa")

    fig.suptitle(
        "Cross-Layer Correlation — PHY K-factor vs NS-3 service-link KPIs\n"
        "(Per-satellite samples from TR 38.811 channel × RT link budget)",
        fontsize=10,
    )

    # ── Panel 1: K-factor vs slot PER ─────────────────────────────────────────
    ax1.scatter(k_arr, per_arr, c=elev_arr, cmap="viridis", s=80,
                edgecolor="#222", lw=0.6)
    for xi, yi, sid in zip(k_arr, per_arr, sid_arr):
        ax1.annotate(f"sat{sid}", (xi, yi), textcoords="offset points",
                     xytext=(5, 4), fontsize=7)

    if len(pairs) >= 2:
        fit = np.polyfit(k_arr, per_arr, 1)
        xs = np.linspace(k_arr.min() - 0.5, k_arr.max() + 0.5, 50)
        ax1.plot(xs, fit[0] * xs + fit[1], "--", color="#d62728", lw=1.4,
                 label=f"linear fit (R={r_k_per:+.2f}, R²={r_k_per**2:.2f})")

    status_per = "PASS" if abs(r_k_per) >= 0.8 else "CHECK"
    ax1.text(0.03, 0.95, f"[{status_per}]  target |R|>0.8",
             transform=ax1.transAxes, fontsize=8,
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="#888",
                       boxstyle="round,pad=0.25"))
    ax1.set_xlabel("K-factor [dB]", fontsize=9)
    ax1.set_ylabel("Slot PER [%]", fontsize=9)
    ax1.set_title("K-factor ↔ Packet error rate", fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right")

    # ── Panel 2: K-factor vs slot one-way delay ───────────────────────────────
    ax2.scatter(k_arr, delay_arr, c=elev_arr, cmap="viridis", s=80,
                edgecolor="#222", lw=0.6)
    for xi, yi, sid in zip(k_arr, delay_arr, sid_arr):
        ax2.annotate(f"sat{sid}", (xi, yi), textcoords="offset points",
                     xytext=(5, 4), fontsize=7)

    if len(pairs) >= 2:
        fit2 = np.polyfit(k_arr, delay_arr, 1)
        xs2  = np.linspace(k_arr.min() - 0.5, k_arr.max() + 0.5, 50)
        ax2.plot(xs2, fit2[0] * xs2 + fit2[1], "--", color="#d62728", lw=1.4,
                 label=f"linear fit (R={r_k_delay:+.2f}, R²={r_k_delay**2:.2f})")

    status_delay = "PASS" if abs(r_k_delay) >= 0.8 else "CHECK"
    ax2.text(0.03, 0.95, f"[{status_delay}]  target |R|>0.8",
             transform=ax2.transAxes, fontsize=8,
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="#888",
                       boxstyle="round,pad=0.25"))
    ax2.set_xlabel("K-factor [dB]", fontsize=9)
    ax2.set_ylabel("Slot one-way delay [ms]", fontsize=9)
    ax2.set_title("K-factor ↔ Slot propagation delay", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="upper right")

    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=plt.Normalize(vmin=elev_arr.min(),
                                                    vmax=elev_arr.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.85, pad=0.02)
    cbar.set_label("Elevation [°]", fontsize=9)

    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[CrossLayer]  Saved -> {out}  "
          f"(R_K↔PER={r_k_per:+.2f}, R_K↔delay={r_k_delay:+.2f})")
    plt.close()
    return out
