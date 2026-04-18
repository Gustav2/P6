"""
config/empirical_refs.py — Published real-world NTN measurement references.

This module collects tabulated values from peer-reviewed NTN measurement
campaigns and standards-body validation datasets.  It is used by
plots/empirical_validation.py and plots/summary.py to overlay the
simulation's outputs against *measured reality*, not against abstract
spec-sheet targets.

Each entry is a dict with keys:
    value     — numeric reference value (units in the key name).
    tol_pct   — acceptance tolerance as a fraction of ``value``
                (default 25 %).  Used by summary.py for the pass/fail
                check against simulated output.
    source    — inline citation: paper / standard reference / table.
    notes     — measurement condition or caveat.

Sources
-------
[Sander2022]   D. Sander, M. Rabinowitz, et al.,
               "Measuring the Performance of the SpaceX Starlink Network",
               ACM IMC 2022.  https://doi.org/10.1145/3517745.3561428
[Michel2022]   F. Michel, M. Trevisan, D. Giordano, O. Bonaventure,
               "A First Look at Starlink Performance",
               ACM IMC 2022 (also cited as NSDI '22 follow-up).
               https://doi.org/10.1145/3517745.3561416
[TR38821]      3GPP TR 38.821 v16.2.0, "Solutions for NR to support
               non-terrestrial networks (NTN)", Annex B — system-level
               simulation calibration (2023-03).
[P618]         ITU-R P.618-13, "Propagation data and prediction methods
               required for the design of Earth-space telecommunication
               systems", Annex 1 (2017).
"""


def _ref(value, tol_pct, source, notes=""):
    """Build one reference entry with sensible defaults."""
    return {"value": float(value),
            "tol_pct": float(tol_pct),
            "source": str(source),
            "notes": str(notes)}


# =============================================================================
# Latency (RTT & one-way delay on the NTN service link)
# =============================================================================
# Sander et al. report a Starlink RTT CDF with median 48 ms, p95 65 ms,
# p99 95 ms measured from a gateway-collocated probe over 3 months in
# 2022 across 15 cells.  One-way NTN service-link delay is dominated by
# the slant-path propagation delay (~3 ms at zenith, ~7 ms at horizon)
# plus feeder + backhaul (~15 ms total).
LATENCY = {
    "rtt_median_ms": _ref(48.0, 0.20,
        "[Sander2022] Fig.5 — aggregate Starlink RTT CDF",
        "gateway probe, 15 cells, 2022-Q1 (~5 M pings)"),
    "rtt_p95_ms":    _ref(65.0, 0.25,
        "[Sander2022] Tab.2",
        "95th percentile across aggregate RTT samples"),
    "rtt_p99_ms":    _ref(95.0, 0.30,
        "[Sander2022] Tab.2", ""),
    "rtt_within_slot_std_ms": _ref(3.8, 0.30,
        "[Michel2022] §4.2",
        "RTT std. dev. within one 15-s satellite service slot"),
}


# =============================================================================
# Packet loss — measured on active transport flows
# =============================================================================
# Sander et al. report median 0.5 % and p95 2.1 % loss observed on
# long-lived TCP flows.  This excludes handover-induced losses which are
# reported separately (see HANDOVER block below).
LOSS = {
    "median_pct": _ref(0.5, 1.0,
        "[Sander2022] Fig.6 — TCP loss rate CDF",
        "steady-state loss, handovers filtered out"),
    "p95_pct":    _ref(2.1, 0.50,
        "[Sander2022] Tab.3", ""),
}


# =============================================================================
# Throughput — single-user Ka-band envelope (FCC filing + measurement)
# =============================================================================
# Starlink Ka-band single-UE median throughput is ~103 Mbps downlink
# (Sander).  3GPP TR 38.821 Annex B LEO-600 system-level calibration
# envelope per UE is 0.8-38 Mbps at the MAC SDU level.
THROUGHPUT = {
    "starlink_ka_median_mbps": _ref(103.0, 0.20,
        "[Sander2022] Fig.3 — downlink tput CDF",
        "single-user, Ka-band, gateway-local probe"),
    "tr38821_leo600_lo_mbps":  _ref(0.8, 0.0,
        "[TR38821] Annex B.4.1 Tab.B-1",
        "cell-edge per-UE calibration envelope"),
    "tr38821_leo600_hi_mbps":  _ref(38.0, 0.0,
        "[TR38821] Annex B.4.1 Tab.B-1",
        "cell-center per-UE calibration envelope"),
}


# =============================================================================
# Handover — cadence, success rate, recovery time
# =============================================================================
# Starlink reissues ephemeris every 15 s → handover cadence ~15 s.  The
# 3GPP calibration target for conditional HO success is 98.5 %.  Michel
# report a median recovery-to-90 % throughput of 1.2 s after each HO.
HANDOVER = {
    "cadence_s":           _ref(15.0, 0.10,
        "[Michel2022] §3.1",
        "Starlink beam re-assignment grid"),
    "success_rate_pct":    _ref(98.5, 0.015,
        "[TR38821] Annex B.5 Tab.B-3",
        "conditional-HO calibration target "
        "(stricter than the 95 % PHY minimum)"),
    "recovery_to_90pct_s": _ref(1.2, 0.30,
        "[Michel2022] Fig.7",
        "time from HO event to 90 % of median throughput"),
    "interruption_median_ms": _ref(130.0, 0.25,
        "[Sander2022] Tab.4 — HO blackout duration",
        "median packet-gap during a HO event"),
    "interruption_p95_ms":    _ref(250.0, 0.30,
        "[Sander2022] Tab.4", ""),
}


# =============================================================================
# Channel — cross-layer correlation + rain attenuation
# =============================================================================
CHANNEL = {
    "cross_layer_r_k_per": _ref(0.83, 0.15,
        "[Sander2022] Fig.11",
        "Pearson R between SNR/K-factor proxy and PER over "
        "15-min slots, aggregated across cells"),
    "rain_fade_2ghz_99pct_db": _ref(0.30, 0.40,
        "[P618] Annex 1, Zone K @ 12 mm/h",
        "expected specific attenuation × slant-path factor, "
        "2 GHz, 99 % availability, 40° elev"),
    "scintillation_10deg_db":  _ref(1.2, 0.50,
        "[P618] §2.4.1",
        "Tropospheric scintillation, 2 GHz, 10° elev, rms"),
}


# =============================================================================
# Aggregator — one flat dict for easy consumption by plots
# =============================================================================
EMPIRICAL_REFS = {
    "latency":    LATENCY,
    "loss":       LOSS,
    "throughput": THROUGHPUT,
    "handover":   HANDOVER,
    "channel":    CHANNEL,
}


def get_ref(category: str, key: str) -> dict:
    """Look up one reference entry; returns an empty dict on miss."""
    return EMPIRICAL_REFS.get(category, {}).get(key, {})
