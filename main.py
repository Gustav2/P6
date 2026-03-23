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

import os
import numpy as np

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
import matplotlib
matplotlib.use("Agg")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Force LLVM (CPU) Mitsuba variant before any sionna.rt import.
# Without this, Sionna tries cuda_ad_mono_polarized first; on CPU-only
# machines the DrJIT GPU init crashes with a fatal GIL segfault.
import mitsuba as mi
if mi.variant() is None:
    mi.set_variant("llvm_ad_mono_polarized")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from config import (
    CARRIER_FREQ_HZ,
    SAT_HEIGHT_M,
    ELEVATION_ANGLE_DEG,
    CODERATE,
    NUM_BITS_PER_SYMBOL,
    FFT_SIZE,
    NUM_OFDM_SYMBOLS,
    PILOT_SYMBOL_IDX,
    SNR_THRESH_DB,
    SIGMOID_SLOPE,
    PROTOCOLS,
)
from ntn_phy          import run_sionna_ber
from rt_sim           import run_ray_tracing
from ntn_ns3          import run_ns3_both_topologies
from topology_diagram import (
    draw_protocol_comparison,
    draw_summary,
    draw_link_budget_waterfall,
    draw_snr_vs_elevation,
    draw_latency_breakdown,
    draw_handover_impact,
    draw_protocol_radar,
    draw_combined_results,
    draw_timeseries,
    draw_fairness,
    draw_profile_breakdown,
)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("  NTN Satellite Simulation")
    print("  Sionna 1.2.1 + OpenNTN (TR38.811) + Sionna RT + NS-3")
    print("=" * 70)

    # ── Part 1: PHY layer BER/BLER ────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Part 1 — PHY layer BER/BLER  (Sionna + OpenNTN)")
    print("─" * 70)

    snr_range   = np.arange(0, 20.5, 0.5, dtype=float)  # 0.5 dB steps for sigmoid fitting
    ber_results = {}
    for sc in ["urban", "dense_urban", "suburban"]:
        ber, bler       = run_sionna_ber(snr_range, sc)
        ber_results[sc] = (ber, bler)
        print(f"  [{sc}]  BER @ 10 dB = {ber[20]:.4f}  BLER @ 10 dB = {bler[20]:.4f}")

    # ── Sigmoid fitting: BER → PER (using "urban" scenario curve) ────────────
    # Convert BER to PER:  PER = 1 - (1 - BER)^(codeword_bits)
    # where codeword_bits is the number of information bits per LDPC codeword.
    #
    # The Sionna ResourceGrid has:
    #   - NUM_OFDM_SYMBOLS total OFDM symbols per slot (= 14 at µ=0)
    #   - len(PILOT_SYMBOL_IDX) pilot symbols (= 2, at indices 2 and 11)
    #   - FFT_SIZE active subcarriers per OFDM symbol (no guard-band stripping
    #     in Sionna's ResourceGrid by default)
    #   - NUM_BITS_PER_SYMBOL bits per data symbol (= 2 for QPSK)
    # Total coded bits per slot:
    #   n = (NUM_OFDM_SYMBOLS - num_pilots) × FFT_SIZE × NUM_BITS_PER_SYMBOL
    #     = (14 - 2) × 128 × 2 = 3072 bits
    # Information bits:
    #   k = n × CODERATE = 3072 × 0.5 = 1536 bits
    # This matches the _n and _k computed in NTNOFDMModel.__init__() in ntn_phy.py
    # (lines 134–135), which also uses rg.num_data_symbols × NUM_BITS_PER_SYMBOL.
    #
    # NOTE: Using FFT_SIZE × CODERATE × NUM_BITS_PER_SYMBOL = 128 (incorrect)
    # would underestimate the codeword by 24×, giving a much shallower BER→PER
    # waterfall and causing the fitted sigmoid slope to be too small.
    num_pilot_syms = len(PILOT_SYMBOL_IDX)                             # = 2
    num_data_syms  = NUM_OFDM_SYMBOLS - num_pilot_syms                 # = 12
    codeword_bits  = num_data_syms * FFT_SIZE * NUM_BITS_PER_SYMBOL    # = 3072

    # Fit sigmoid(snr, thresh, slope) = 1 / (1 + exp(slope*(snr - thresh)))
    # to the PER vs Eb/N0 curve, then pass the fitted params to NS-3.
    print("\n  Fitting BER→PER sigmoid to urban Sionna LDPC BER curve ...")
    fitted_snr_thresh    = SNR_THRESH_DB
    fitted_sigmoid_slope = SIGMOID_SLOPE
    try:
        from scipy.optimize import curve_fit

        def _sigmoid(snr, thresh, slope):
            return 1.0 / (1.0 + np.exp(slope * (snr - thresh)))

        ber_urban, _ = ber_results["urban"]
        per_urban = 1.0 - (1.0 - np.clip(ber_urban, 0.0, 1.0 - 1e-9)) ** codeword_bits

        # Only fit over points where PER is in (0.01, 0.99) — the transition
        # region carries the most information about the sigmoid shape.
        mask = (per_urban > 0.01) & (per_urban < 0.99)
        if mask.sum() >= 2:
            popt, _ = curve_fit(
                _sigmoid,
                snr_range[mask],
                per_urban[mask],
                p0=[SNR_THRESH_DB, SIGMOID_SLOPE],
                bounds=([0.0, 0.01], [30.0, 5.0]),
                maxfev=10000,
            )
            fitted_snr_thresh    = float(popt[0])
            fitted_sigmoid_slope = float(popt[1])
            print(f"  Sigmoid fit (urban):  "
                  f"snr_thresh={fitted_snr_thresh:.2f} dB  "
                  f"slope={fitted_sigmoid_slope:.4f} /dB")
            print(f"  Config defaults:      "
                  f"snr_thresh={SNR_THRESH_DB:.2f} dB  "
                  f"slope={SIGMOID_SLOPE:.4f} /dB")
        else:
            print(f"  Warning: not enough PER transition points for fitting "
                  f"({mask.sum()} valid pts).  Using config defaults.")
    except Exception as exc:
        print(f"  Warning: sigmoid fitting failed ({exc}).  "
              f"Using config defaults (snr_thresh={SNR_THRESH_DB}, "
              f"slope={SIGMOID_SLOPE}).")

    # ── Part 2: Ray tracing  (MUST run before NS-3) ───────────────────────────
    print("\n" + "─" * 70)
    print("  Part 2 — Ray tracing  (Sionna RT, Munich scene)")
    print("─" * 70)

    channel_stats = run_ray_tracing()

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
    from config import USE_BASE_STATIONS, NUM_STATIONARY_CLIENTS, NUM_MOVING_CLIENTS
    _topo = ("indirect: Phone→gNB→AccessSat→ISL→BenchmarkSat→GS→Server"
             if USE_BASE_STATIONS
             else "direct: Phone→AccessSat→ISL→BenchmarkSat→GS→Server")
    print(f"  Topology : {_topo}")
    print(f"  Clients  : {NUM_STATIONARY_CLIENTS} stationary + "
          f"{NUM_MOVING_CLIENTS} moving  "
          f"(total {NUM_STATIONARY_CLIENTS + NUM_MOVING_CLIENTS})")
    print("─" * 70)

    direct_results, _ = run_ns3_both_topologies(
        channel_stats, scenario="urban",
        snr_thresh_db=fitted_snr_thresh,
        sigmoid_slope=fitted_sigmoid_slope,
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Plots")
    print("─" * 70)

    draw_protocol_comparison(direct_results, out="output/ntn_protocol_comparison.png")
    print("  [1/8]  output/ntn_protocol_comparison.png")

    draw_summary(ber_results, direct_results, snr_range=snr_range, out="output/ntn_summary.png")
    print("  [2/8]  output/ntn_summary.png")

    draw_link_budget_waterfall(channel_stats, out="output/ntn_link_budget_waterfall.png")
    print("  [3/8]  output/ntn_link_budget_waterfall.png")

    draw_snr_vs_elevation(channel_stats, out="output/ntn_snr_vs_elevation.png")
    print("  [4/8]  output/ntn_snr_vs_elevation.png")

    draw_latency_breakdown(direct_results, out="output/ntn_latency_breakdown.png")
    print("  [5/8]  output/ntn_latency_breakdown.png")

    draw_handover_impact(direct_results, out="output/ntn_handover_impact.png")
    print("  [6/8]  output/ntn_handover_impact.png")

    draw_protocol_radar(direct_results, out="output/ntn_protocol_radar.png")
    print("  [7/8]  output/ntn_protocol_radar.png")

    draw_combined_results(direct_results, out="output/ntn_results.png")
    print("  [8/8]  output/ntn_results.png")

    draw_timeseries(direct_results, out="output/ntn_timeseries.png")
    print("  [9/11] output/ntn_timeseries.png")

    draw_fairness(direct_results, out="output/ntn_fairness.png")
    print("  [10/11] output/ntn_fairness.png")

    draw_profile_breakdown(direct_results, out="output/ntn_profile_breakdown.png")
    print("  [11/11] output/ntn_profile_breakdown.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Simulation complete.")
    print()
    print("  Output files (output/ subdirectory):")
    print("    output/ntn_protocol_comparison.png    — 4-panel protocol comparison")
    print("    output/ntn_summary.png                — BER/BLER + NS-3 combined figure")
    print("    output/ntn_link_budget_waterfall.png  — per-satellite link budget waterfall")
    print("    output/ntn_snr_vs_elevation.png       — SNR vs elevation + PER sigmoid")
    print("    output/ntn_latency_breakdown.png      — per-hop stacked latency bars (NTN/ISL/backhaul)")
    print("    output/ntn_handover_impact.png        — per-slot throughput bars")
    print("    output/ntn_protocol_radar.png         — 5-axis radar chart")
    print("    output/ntn_results.png                — 3-panel combined summary")
    print("    output/ntn_timeseries.png             — per-second throughput with HO gap markers")
    print("    output/ntn_fairness.png               — Jain's fairness index per protocol")
    print("    output/ntn_profile_breakdown.png      — throughput/loss by traffic profile")
    print("    output/ntn_rt_paths_sat<N>.png        — RT paths per satellite")
    print("    output/ntn_rt_radiomap.png            — composite radio map")
    print("=" * 70)

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
