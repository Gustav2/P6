"""
main.py — NTN Simulation entry point
=====================================
Orchestrates the three simulation parts in the correct order:

  Part 1 — ntn_phy.py         : Sionna 1.2.1 + OpenNTN  BER/BLER link sim
  Part 2 — rt_sim.py          : Sionna RT ray tracing (Munich scene)
                                  → returns channel_stats for NS-3
  Part 3 — ntn_ns3.py         : NS-3 multi-protocol packet simulation
                                  (uses RT channel_stats for link budget)
                                  Runs 5 protocols x 2 topologies (direct / indirect)
  Plots  — topology_diagram.py: topology illustration + all comparison charts

Pipeline
--------
  run_sionna_ber()                          → ber_results
  run_ray_tracing()                         → channel_stats   ← MUST come before NS-3
  run_ns3_both_topologies(stats)            → direct_results, indirect_results
  draw_topology()
  draw_protocol_comparison(direct_results)
  draw_summary(ber_results, direct_results)
  draw_throughput_over_time(direct, indirect)
  draw_topology_comparison()
  draw_direct_vs_indirect(direct, indirect)
  draw_link_budget_waterfall(channel_stats)
  draw_snr_vs_elevation(channel_stats)
  draw_latency_breakdown(direct, indirect)
  draw_handover_impact(direct, indirect)
  draw_protocol_radar(direct, indirect)
  draw_combined_results(direct, indirect)

Usage
-----
  python main.py

Output files
------------
All PNG output files are written to the output/ subdirectory.

  output/ntn_topology.png               — Scenario topology diagram
  output/ntn_protocol_comparison.png    — Grouped bar chart (latency / tput / loss)
  output/ntn_summary.png                — Five-panel BER + NS-3 combined figure
  output/ntn_throughput_over_time.png   — Per-protocol throughput vs time (direct + indirect)
  output/ntn_network_illustration.png   — Detailed NTN network diagram
  output/ntn_ue_satellite.png           — UE-to-satellite street scene
  output/ntn_topology_comparison.png    — Side-by-side direct vs indirect architecture
  output/ntn_direct_vs_indirect.png     — Grouped bars: 3 metrics x 5 protocols x 2 topologies
  output/ntn_link_budget_waterfall.png  — Per-satellite link budget waterfall
  output/ntn_snr_vs_elevation.png       — SNR vs elevation + PER sigmoid
  output/ntn_latency_breakdown.png      — Per-hop stacked latency bars
  output/ntn_handover_impact.png        — Per-slot throughput (TCP collapse at Sat 2)
  output/ntn_protocol_radar.png         — 5-axis radar chart (direct + indirect)
  output/ntn_results.png                — 6-panel combined results summary
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
    PROTOCOLS,
)
from ntn_phy          import run_sionna_ber
from rt_sim           import run_ray_tracing
from ntn_ns3          import run_ns3_both_topologies
from topology_diagram import (
    draw_topology,
    draw_protocol_comparison,
    draw_summary,
    draw_throughput_over_time,
    draw_network_illustration,
    draw_ue_satellite_scene,
    draw_topology_comparison,
    draw_direct_vs_indirect,
    draw_link_budget_waterfall,
    draw_snr_vs_elevation,
    draw_latency_breakdown,
    draw_handover_impact,
    draw_protocol_radar,
    draw_combined_results,
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

    snr_range   = np.arange(0, 22, 2, dtype=float)
    ber_results = {}
    for sc in ["urban", "dense_urban", "suburban"]:
        ber, bler       = run_sionna_ber(snr_range, sc)
        ber_results[sc] = (ber, bler)
        print(f"  [{sc}]  BER @ 10 dB = {ber[5]:.4f}  BLER @ 10 dB = {bler[5]:.4f}")

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
    print("  Topologies: direct (Phone->Sat->GS->Server)  +  "
          "indirect (Phone->gNB->Sat->GS->Server)")
    print("─" * 70)

    direct_results, indirect_results = run_ns3_both_topologies(
        channel_stats, scenario="urban"
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Plots")
    print("─" * 70)

    draw_topology(out="output/ntn_topology.png")
    print("  [1/14]  output/ntn_topology.png")

    draw_protocol_comparison(direct_results, out="output/ntn_protocol_comparison.png")
    print("  [2/14]  output/ntn_protocol_comparison.png")

    draw_summary(ber_results, direct_results, out="output/ntn_summary.png")
    print("  [3/14]  output/ntn_summary.png")

    draw_throughput_over_time(direct_results, indirect_results,
                              out="output/ntn_throughput_over_time.png")
    print("  [4/14]  output/ntn_throughput_over_time.png")

    draw_network_illustration(out="output/ntn_network_illustration.png")
    print("  [5/14]  output/ntn_network_illustration.png")

    draw_ue_satellite_scene(out="output/ntn_ue_satellite.png")
    print("  [6/14]  output/ntn_ue_satellite.png")

    draw_topology_comparison(out="output/ntn_topology_comparison.png")
    print("  [7/14]  output/ntn_topology_comparison.png")

    draw_direct_vs_indirect(direct_results, indirect_results,
                            out="output/ntn_direct_vs_indirect.png")
    print("  [8/14]  output/ntn_direct_vs_indirect.png")

    draw_link_budget_waterfall(channel_stats, out="output/ntn_link_budget_waterfall.png")
    print("  [9/14]  output/ntn_link_budget_waterfall.png")

    draw_snr_vs_elevation(channel_stats, out="output/ntn_snr_vs_elevation.png")
    print("  [10/14] output/ntn_snr_vs_elevation.png")

    draw_latency_breakdown(direct_results, indirect_results,
                           out="output/ntn_latency_breakdown.png")
    print("  [11/14] output/ntn_latency_breakdown.png")

    draw_handover_impact(direct_results, indirect_results,
                         out="output/ntn_handover_impact.png")
    print("  [12/14] output/ntn_handover_impact.png")

    draw_protocol_radar(direct_results, indirect_results,
                        out="output/ntn_protocol_radar.png")
    print("  [13/14] output/ntn_protocol_radar.png")

    draw_combined_results(direct_results, indirect_results,
                          out="output/ntn_results.png")
    print("  [14/14] output/ntn_results.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Simulation complete.")
    print()
    print("  Output files (output/ subdirectory):")
    print("    output/ntn_topology.png               — topology diagram")
    print("    output/ntn_protocol_comparison.png    — protocol comparison bar chart")
    print("    output/ntn_summary.png                — BER/BLER + NS-3 combined figure")
    print("    output/ntn_throughput_over_time.png   — per-protocol throughput vs time")
    print("    output/ntn_network_illustration.png   — detailed NTN network diagram")
    print("    output/ntn_ue_satellite.png           — UE-to-satellite street scene")
    print("    output/ntn_topology_comparison.png    — direct vs indirect architecture")
    print("    output/ntn_direct_vs_indirect.png     — grouped bars: 3 metrics x 2 topologies")
    print("    output/ntn_link_budget_waterfall.png  — per-satellite link budget waterfall")
    print("    output/ntn_snr_vs_elevation.png       — SNR vs elevation + PER sigmoid")
    print("    output/ntn_latency_breakdown.png      — per-hop stacked latency bars")
    print("    output/ntn_handover_impact.png        — per-slot throughput bars")
    print("    output/ntn_protocol_radar.png         — 5-axis radar chart")
    print("    output/ntn_results.png                — 6-panel combined summary")
    print("    output/ntn_rt_paths_sat<N>.png        — RT paths per satellite")
    print("    output/ntn_rt_radiomap.png            — composite radio map")
    print("=" * 70)

    # ── Protocol results table ────────────────────────────────────────────────
    for topology_label, results in [("Direct", direct_results),
                                     ("Indirect", indirect_results)]:
        print()
        print(f"  [{topology_label} topology]")
        print(f"  {'Protocol':<16}  {'Latency (ms)':>12}  {'Tput (kbps)':>12}"
              f"  {'Loss (%)':>9}  {'Handovers':>10}")
        print("  " + "─" * 64)
        for r in results:
            print(f"  {r['label']:<16}  {r['mean_delay_ms']:>12.1f}"
                  f"  {r['throughput_kbps']:>12.0f}"
                  f"  {r['loss_pct']:>9.2f}"
                  f"  {r.get('handovers', 0):>10}")


if __name__ == "__main__":
    main()
