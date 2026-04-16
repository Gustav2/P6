"""
main.py — NTN Simulation entry point
=====================================
Orchestrates the three simulation parts in the correct order:

  Part 1 — ntn_phy.py         : Sionna 1.2.1 + OpenNTN  BER/BLER link sim
  Part 2 — rt_sim.py          : Sionna RT ray tracing (Munich scene)
                                  → returns channel_stats for NS-3
  Part 3 — ntn_ns3.py         : NS-3 multi-protocol packet simulation
                                  (uses RT channel_stats for link budget)
  Plots  — topology_diagram.py: all comparison and analysis charts

Pipeline
--------
  run_sionna_ber()                          → ber_results
  run_ray_tracing()                         → channel_stats   ← MUST come before NS-3
  run_ns3_both_topologies(stats)            → direct_results, _
  draw_protocol_comparison(direct_results)
  draw_summary(ber_results, direct_results)
  draw_link_budget_waterfall(channel_stats)
  draw_snr_vs_elevation(channel_stats)
  draw_latency_breakdown(direct_results)
  draw_handover_impact(direct_results)
  draw_protocol_radar(direct_results)
  draw_combined_results(direct_results)

Usage
-----
  python main.py

Output files
------------
All PNG output files are written to the output/ subdirectory.

  output/ntn_protocol_comparison.png    — Grouped bar chart (latency / tput / loss)
  output/ntn_summary.png                — Five-panel BER + NS-3 combined figure
  output/ntn_link_budget_waterfall.png  — Per-satellite link budget waterfall
  output/ntn_snr_vs_elevation.png       — SNR vs elevation + PER sigmoid
  output/ntn_latency_breakdown.png      — Per-hop stacked latency bars
  output/ntn_handover_impact.png        — Per-slot throughput bars
  output/ntn_protocol_radar.png         — 5-axis radar chart
  output/ntn_results.png                — 3-panel combined results summary
  output/ntn_rt_paths_sat<N>.png        — Ray traced paths per satellite
  output/ntn_rt_radiomap.png            — Composite radio map
"""

import hashlib
import os
import pickle
import time
import numpy as np

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
import matplotlib
matplotlib.use("Agg")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Mitsuba's cuda_ad_mono_polarized variant requires OptiX for scene/BVH work.
# Check for libnvoptix before committing to the CUDA variant; fall back to the
# LLVM JIT variant (CPU) when OptiX is absent (e.g. Jetson Orin without OptiX).
import ctypes.util as _ctypes_util
import mitsuba as mi
if mi.variant() is None:
    if _ctypes_util.find_library("nvoptix"):
        mi.set_variant("cuda_ad_mono_polarized")
    else:
        mi.set_variant("llvm_ad_mono_polarized")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# Maximise CPU thread usage for TF PHY simulation on the 8-core Jetson CPU.
# TF on Jetson has no GPU support compiled in, so all compute runs on CPU.
_n_cores = os.cpu_count() or 8
tf.config.threading.set_inter_op_parallelism_threads(_n_cores)
tf.config.threading.set_intra_op_parallelism_threads(_n_cores)

from config import (
    CARRIER_FREQ_HZ,
    SUBCARRIER_SPACING,
    FFT_SIZE,
    NUM_OFDM_SYMBOLS,
    CP_LENGTH,
    PILOT_SYMBOL_IDX,
    NUM_BITS_PER_SYMBOL,
    CODERATE,
    BATCH_SIZE,
    SAT_HEIGHT_M,
    ELEVATION_ANGLE_DEG,
    SNR_THRESH_DB,
    SIGMOID_SLOPE,
    PROTOCOLS,
    CONSTELLATION_TOTAL_SATS,
    VISIBLE_SATELLITES_PER_PASS,
    NUM_PEDESTRIAN_MOVING_CLIENTS,
    NUM_VEHICULAR_MOVING_CLIENTS,
    PEDESTRIAN_SPEED_MIN_MS,
    PEDESTRIAN_SPEED_MAX_MS,
    VEHICULAR_SPEED_MIN_MS,
    VEHICULAR_SPEED_MAX_MS,
    RT_UE_SAMPLE_POSITIONS,
)
from ntn_phy          import run_sionna_ber
from rt_sim           import run_ray_tracing
from ntn_ns3          import run_ns3_both_topologies
from topology_diagram import (
    draw_ber_bler,
    draw_protocol_comparison,
    draw_link_budget_waterfall,
    draw_snr_vs_elevation,
    draw_latency_breakdown,
    draw_handover_impact,
    draw_handover_schedule,
    draw_timeseries,
    draw_fairness,
    draw_profile_breakdown,
    draw_timing_breakdown,
)


# =============================================================================
# PHY result cache
# =============================================================================
# The BER/BLER curves are deterministic for fixed config.  Caching them
# saves the full PHY stage (typically 3-7 min) on repeated runs when no
# PHY-related parameters have changed.

def _phy_cache_path() -> str:
    """Return cache file path based on a hash of all PHY-relevant config values."""
    key = hashlib.md5(str({
        "freq":   CARRIER_FREQ_HZ,
        "scs":    SUBCARRIER_SPACING,
        "fft":    FFT_SIZE,
        "syms":   NUM_OFDM_SYMBOLS,
        "cp":     CP_LENGTH,
        "pilots": tuple(PILOT_SYMBOL_IDX),
        "bps":    NUM_BITS_PER_SYMBOL,
        "rate":   CODERATE,
        "batch":  BATCH_SIZE,
        "height": SAT_HEIGHT_M,
        "elev":   ELEVATION_ANGLE_DEG,
    }).encode()).hexdigest()[:10]
    return f"{OUTPUT_DIR}/.phy_cache_{key}.pkl"


def _load_phy_cache() -> tuple:
    """Return (data, path). data is None on cache miss."""
    path = _phy_cache_path()
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f), path
    return None, path


def _save_phy_cache(data: dict) -> None:
    with open(_phy_cache_path(), "wb") as f:
        pickle.dump(data, f)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("  NTN Satellite Simulation")
    print("  Sionna 1.2.1 + OpenNTN (TR38.811) + Sionna RT + NS-3")
    print(f"  Mitsuba variant : {mi.variant()}")
    print(f"  TF threads      : {_n_cores} cores")
    print("=" * 70)

    t_start = time.perf_counter()
    timing  = {}   # stage → elapsed seconds

    # ── Part 1: PHY layer BER/BLER ────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Part 1 — PHY layer BER/BLER  (Sionna + OpenNTN)")
    print("─" * 70)

    # 1 dB steps give 21 SNR points — sufficient resolution for sigmoid fitting
    # and 2× faster than the original 0.5 dB grid (41 points).
    snr_range   = np.arange(0, 21.0, 1.0, dtype=float)

    t0 = time.perf_counter()
    cached_phy, cache_path = _load_phy_cache()
    if cached_phy is not None:
        ber_results = cached_phy
        print(f"  [PHY cache hit]  Loaded from {cache_path}  (delete to re-run)")
        for sc, (ber, bler) in ber_results.items():
            print(f"  [{sc}]  BER @ 10 dB = {ber[20]:.4f}  BLER @ 10 dB = {bler[20]:.4f}")
        timing["PHY (cached)"] = time.perf_counter() - t0
    else:
        ber_results = {}
        for sc in ["urban", "dense_urban", "suburban"]:
            ber, bler       = run_sionna_ber(snr_range, sc)
            ber_results[sc] = (ber, bler)
            print(f"  [{sc}]  BER @ 10 dB = {ber[20]:.4f}  BLER @ 10 dB = {bler[20]:.4f}")
        _save_phy_cache(ber_results)
        print(f"  [PHY cache saved]  {cache_path}")
        timing["PHY (Sionna)"] = time.perf_counter() - t0

    # ── Sigmoid fitting (using "urban" scenario BLER curve) ──────────────────
    # Fit sigmoid(snr, thresh, slope) = 1 / (1 + exp(slope*(snr - thresh)))
    # to the PER vs Eb/N0 curve, then pass the fitted params to NS-3.
    print("\n  Fitting BER→PER sigmoid to urban Sionna LDPC BER curve ...")
    fitted_snr_thresh    = SNR_THRESH_DB
    fitted_sigmoid_slope = SIGMOID_SLOPE
    try:
        from scipy.optimize import curve_fit

        def _sigmoid(snr, thresh, slope):
            return 1.0 / (1.0 + np.exp(slope * (snr - thresh)))

        ber_urban, bler_urban = ber_results["urban"]

        # Use BLER (block error rate) directly as the PER target.
        # Sionna measures BLER = fraction of codewords with ≥1 bit error,
        # which is exactly the packet error probability for one packet per
        # codeword.  This is more reliable than the analytic BER→PER
        # transformation  1-(1-BER)^n  which collapses the waterfall so
        # steeply that almost no points land in the sigmoid transition zone.
        per_fit = np.array(bler_urban, dtype=float)

        # Enforce monotonic decrease (BLER should fall as Eb/N0 rises).
        # Monte Carlo noise at low error counts can cause small bumps;
        # cumulative-minimum from left to right clips them.
        per_fit = np.minimum.accumulate(per_fit)

        # Fit over the transition region: 0.001 < BLER < 0.999.
        # The lower bound is relaxed vs the old 0.01 so that the low-BLER
        # tail of the LDPC waterfall (which is resolved at BATCH_SIZE=512)
        # contributes to the slope estimate.
        mask = (per_fit > 0.001) & (per_fit < 0.999)
        if mask.sum() >= 2:
            popt, _ = curve_fit(
                _sigmoid,
                snr_range[mask],
                per_fit[mask],
                p0=[SNR_THRESH_DB, SIGMOID_SLOPE],
                bounds=([0.0, 0.01], [20.0, 10.0]),
                maxfev=10000,
            )
            fitted_snr_thresh    = float(popt[0])
            fitted_sigmoid_slope = float(popt[1])

            # Sanity-check: if the threshold hit a bound, the fit is
            # degenerate — fall back to the physically-grounded defaults.
            if not (0.5 < fitted_snr_thresh < 19.5):
                print(f"  Warning: fitted snr_thresh={fitted_snr_thresh:.2f} dB "
                      f"hit boundary — using config defaults.")
                fitted_snr_thresh    = SNR_THRESH_DB
                fitted_sigmoid_slope = SIGMOID_SLOPE
            else:
                print(f"  Sigmoid fit (urban):  "
                      f"snr_thresh={fitted_snr_thresh:.2f} dB  "
                      f"slope={fitted_sigmoid_slope:.4f} /dB")
            print(f"  Config defaults:      "
                  f"snr_thresh={SNR_THRESH_DB:.2f} dB  "
                  f"slope={SIGMOID_SLOPE:.4f} /dB")
        else:
            print(f"  Warning: not enough BLER transition points for fitting "
                  f"({mask.sum()} valid pts).  Using config defaults.")
    except Exception as exc:
        print(f"  Warning: sigmoid fitting failed ({exc}).  "
              f"Using config defaults (snr_thresh={SNR_THRESH_DB}, "
              f"slope={SIGMOID_SLOPE}).")

    # ── Part 2: Ray tracing  (MUST run before NS-3) ───────────────────────────
    print("\n" + "─" * 70)
    print("  Part 2 — Ray tracing  (Sionna RT, Munich scene)")
    print("─" * 70)

    t0 = time.perf_counter()
    channel_stats = run_ray_tracing()
    timing["RT (Sionna RT)"] = time.perf_counter() - t0

    print(f"\n  Channel stats ({len(channel_stats)} satellites):")
    for s in channel_stats:
        print(f"    sat{s['sat_id']}  elev={s['elevation_deg']:.1f}°  "
              f"paths={s['num_paths']}  "
              f"mean_gain={s['mean_path_gain_db']:.1f} dB  "
              f"delay_spread={s['delay_spread_ns']:.1f} ns")

    # ── Part 3: Network simulation per protocol ───────────────────────────────
    print("\n" + "─" * 70)
    print("  Part 3 — NS-3 multi-protocol simulation")
    print(f"  Protocols: {[p['label'] for p in PROTOCOLS]}")
    from config import NUM_STATIONARY_CLIENTS, NUM_MOVING_CLIENTS
    print("  Topology : direct: Phone→Satellite→Server")
    print(f"  Clients  : {NUM_STATIONARY_CLIENTS} stationary + "
          f"{NUM_MOVING_CLIENTS} moving  "
          f"(total {NUM_STATIONARY_CLIENTS + NUM_MOVING_CLIENTS})")
    print(f"  Mobility : pedestrians={NUM_PEDESTRIAN_MOVING_CLIENTS} "
          f"({PEDESTRIAN_SPEED_MIN_MS:.1f}-{PEDESTRIAN_SPEED_MAX_MS:.1f} m/s), "
          f"vehicular={NUM_VEHICULAR_MOVING_CLIENTS} "
          f"({VEHICULAR_SPEED_MIN_MS:.1f}-{VEHICULAR_SPEED_MAX_MS:.1f} m/s)")
    print(f"  Constell.: total≈{CONSTELLATION_TOTAL_SATS} sats, "
          f"sampled-visible={VISIBLE_SATELLITES_PER_PASS}")
    print(f"  RT UEs   : {len(RT_UE_SAMPLE_POSITIONS)} sample points")
    print("─" * 70)

    t0 = time.perf_counter()
    direct_results, _ = run_ns3_both_topologies(
        channel_stats, scenario="urban",
        snr_thresh_db=fitted_snr_thresh,
        sigmoid_slope=fitted_sigmoid_slope,
    )
    timing["NS-3 (4 protocols)"] = time.perf_counter() - t0

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Plots")
    print("─" * 70)

    t0 = time.perf_counter()
    draw_ber_bler(ber_results, snr_range=snr_range, out="output/ntn_ber_bler.png")
    print("  [1/10] output/ntn_ber_bler.png")

    draw_protocol_comparison(direct_results, out="output/ntn_protocol_comparison.png")
    print("  [2/10] output/ntn_protocol_comparison.png")

    draw_link_budget_waterfall(channel_stats, out="output/ntn_link_budget_waterfall.png")
    print("  [3/10] output/ntn_link_budget_waterfall.png")

    draw_snr_vs_elevation(channel_stats, out="output/ntn_snr_vs_elevation.png")
    print("  [4/10] output/ntn_snr_vs_elevation.png")

    draw_latency_breakdown(direct_results, out="output/ntn_latency_breakdown.png")
    print("  [5/10] output/ntn_latency_breakdown.png")

    draw_handover_impact(direct_results, out="output/ntn_handover_impact.png")
    print("  [6/10] output/ntn_handover_impact.png")

    draw_handover_schedule(direct_results, out="output/ntn_handover_schedule.png")
    print("  [7/10] output/ntn_handover_schedule.png")

    draw_timeseries(direct_results, out="output/ntn_timeseries.png")
    print("  [8/10] output/ntn_timeseries.png")

    draw_fairness(direct_results, out="output/ntn_fairness.png")
    print("  [9/10] output/ntn_fairness.png")

    draw_profile_breakdown(direct_results, out="output/ntn_profile_breakdown.png")
    print("  [10/10] output/ntn_profile_breakdown.png")
    timing["Plotting"] = time.perf_counter() - t0
    timing["Total"] = time.perf_counter() - t_start

    draw_timing_breakdown(timing, mi.variant(), out="output/ntn_timing.png")
    print("  [+]   output/ntn_timing.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Simulation complete.")
    print()
    print("  Output files (output/ subdirectory):")
    print("    output/ntn_ber_bler.png               — BER/BLER vs Eb/N0 (Sionna + OpenNTN)")
    print("    output/ntn_protocol_comparison.png    — 4-panel protocol comparison")
    print("    output/ntn_link_budget_waterfall.png  — per-satellite link budget waterfall")
    print("    output/ntn_snr_vs_elevation.png       — SNR vs elevation + PER sigmoid")
    print("    output/ntn_latency_breakdown.png      — per-hop latency breakdown (NTN + overhead)")
    print("    output/ntn_handover_impact.png        — per-slot throughput bars")
    print("    output/ntn_handover_schedule.png      — Gantt timeline of satellite service slots")
    print("    output/ntn_timeseries.png             — per-second throughput with HO gap markers")
    print("    output/ntn_fairness.png               — Jain's fairness index per protocol")
    print("    output/ntn_profile_breakdown.png      — throughput/loss by traffic profile")
    print("    output/ntn_timing.png                 — stage-by-stage runtime breakdown")
    print("    output/ntn_rt_paths_sat<N>.png        — RT paths per satellite")
    print("    output/ntn_rt_radiomap.png            — composite radio map")
    print("=" * 70)

    print()
    print("  Runtime breakdown:")
    for stage, t in timing.items():
        if stage != "Total":
            print(f"    {stage:<22s}: {t:6.1f} s")
    print(f"  {'Total':<22s}: {timing['Total']:6.1f} s")

    # ── Protocol results table ────────────────────────────────────────────────
    print()
    print(f"  {'Protocol':<16}  {'Latency (ms)':>12}  {'Tput (kbps)':>12}"
          f"  {'Loss (%)':>9}  {'Jitter (ms)':>12}  {'Fairness':>9}  {'Handovers':>10}")
    print("  " + "─" * 85)
    for r in direct_results:
        print(f"  {r['label']:<16}  {r['mean_delay_ms']:>12.1f}"
              f"  {r['throughput_kbps']:>12.0f}"
              f"  {r['loss_pct']:>9.2f}"
              f"  {r.get('jitter_ms', 0.0):>12.2f}"
              f"  {r.get('fairness_index', 0.0):>9.4f}"
              f"  {r.get('handovers', 0):>10}")


if __name__ == "__main__":
    main()
