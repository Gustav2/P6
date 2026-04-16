"""
plots/timing.py — Pipeline timing breakdown
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw_timing_breakdown(timing: dict, variant: str,
                          out: str = "output/ntn_timing.png") -> str:
    """Horizontal bar chart showing runtime per pipeline stage.

    Parameters
    ----------
    timing  : dict mapping stage label → elapsed seconds.
              Must contain a "Total" key; all other keys are plotted as stages.
    variant : Mitsuba variant string (e.g. "cuda_ad_mono_polarized").
    out     : Output file path.
    """
    stages = [k for k in timing if k != "Total"]
    times  = [timing[k] for k in stages]
    total  = timing.get("Total", sum(times))

    # Colours per stage
    palette = {
        "PHY (cached)":    "#4e9af1",
        "PHY (Sionna)":    "#1a5fb4",
        "RT (Sionna RT)":  "#e67e22",
        "NS-3 (4 protocols)": "#27ae60",
        "Plotting":        "#8e44ad",
    }
    colors = [palette.get(s, "#95a5a6") for s in stages]

    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.55 * len(stages) + 1.5)))

    y_pos = range(len(stages))
    bars  = ax.barh(list(y_pos), times, color=colors,
                    edgecolor="#333", linewidth=0.6, height=0.55)

    # Annotate with seconds and percentage
    for bar, t in zip(bars, times):
        pct = 100.0 * t / total if total > 0 else 0.0
        label = f"  {t:.1f} s  ({pct:.0f}%)"
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left", fontsize=9)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(stages, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Wall-clock time [s]", fontsize=9)

    # GPU/CPU badge in title
    accel = "GPU" if "cuda" in (variant or "").lower() else "CPU"
    ax.set_title(
        f"Pipeline Stage Runtimes — 5G-NTN Simulation\n"
        f"Mitsuba variant: {variant or 'unknown'}  |  RT accelerator: {accel}  |  "
        f"Total: {total:.1f} s",
        fontsize=9,
    )

    ax.set_xlim(0, max(times) * 1.35 if times else 1)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[Timing]  Saved -> {out}")
    plt.close()
    return out
