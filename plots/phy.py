"""
plots/phy.py — BER/BLER curves
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SAT_HEIGHT_M, CARRIER_FREQ_HZ


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
