"""
main.py — NTN Simulation entry point
=====================================
Orchestrates the three simulation parts in the correct order:

  Part 1 — ntn_phy.py         : Sionna 1.2.1 + OpenNTN  BER/BLER link sim
  Part 2 — rt_sim.py          : Sionna RT ray tracing (Munich scene)
                                  → returns channel_stats for NS-3
  Part 3 — ntn_ns3.py         : NS-3 multi-protocol packet simulation
                                  (uses RT channel_stats for link budget)
  Plots  — topology_diagram.py: topology illustration + protocol bar charts

Pipeline
--------
  run_sionna_ber()               → ber_results
  run_ray_tracing()              → channel_stats   ← MUST come before NS-3
  run_ns3_protocol_suite(stats)  → ns3_results
  draw_topology()
  draw_protocol_comparison()
  draw_summary(ber_results, ns3_results)

Usage
-----
  python main.py

Output files
------------
  ntn_topology.png             — Scenario topology diagram
  ntn_protocol_comparison.png  — Grouped bar chart (latency / tput / loss)
  ntn_summary.png              — Five-panel BER + NS-3 combined figure
  ntn_rt_paths_sat<N>.png      — Ray traced paths per satellite
  ntn_rt_radiomap.png          — Composite radio map
"""

import os
import numpy as np
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
from ntn_ns3          import run_ns3_protocol_suite
from topology_diagram import (draw_topology, draw_protocol_comparison,
                               draw_summary, draw_throughput_over_time,
                               draw_network_illustration, draw_ue_satellite_scene)


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
    print("─" * 70)

    ns3_results = run_ns3_protocol_suite(channel_stats, scenario="urban")

    # ── Topology diagram ──────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Plots — topology diagram + protocol comparison")
    print("─" * 70)

    draw_topology(out="ntn_topology.png")
    draw_protocol_comparison(ns3_results, out="ntn_protocol_comparison.png")
    draw_summary(ber_results, ns3_results, out="ntn_summary.png")
    draw_throughput_over_time(ns3_results, out="ntn_throughput_over_time.png")
    draw_network_illustration(out="ntn_network_illustration.png")
    draw_ue_satellite_scene(out="ntn_ue_satellite.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Simulation complete.")
    print()
    print("  Output files:")
    print("    ntn_topology.png             — topology diagram")
    print("    ntn_protocol_comparison.png  — protocol comparison bar chart")
    print("    ntn_summary.png              — BER/BLER + NS-3 combined figure")
    print("    ntn_throughput_over_time.png — per-protocol throughput vs time")
    print("    ntn_network_illustration.png — detailed NTN network diagram")
    print("    ntn_ue_satellite.png         — UE-to-satellite street scene")
    print("    ntn_rt_paths_sat<N>.png      — RT paths per satellite")
    print("    ntn_rt_radiomap.png          — composite radio map")
    print("=" * 70)

    # ── Protocol results table ────────────────────────────────────────────────
    print()
    print(f"  {'Protocol':<16}  {'Latency (ms)':>12}  {'Tput (kbps)':>12}"
          f"  {'Loss (%)':>9}  {'Handovers':>10}")
    print("  " + "─" * 64)
    for r in ns3_results:
        print(f"  {r['label']:<16}  {r['mean_delay_ms']:>12.1f}"
              f"  {r['throughput_kbps']:>12.0f}"
              f"  {r['loss_pct']:>9.2f}"
              f"  {r.get('handovers', 0):>10}")


if __name__ == "__main__":
    main()
