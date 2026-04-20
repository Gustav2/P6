"""
ntn_ns3.py — Part 2: NS-3 packet-level 5G-NTN network simulation
=================================================================
Simulates the end-to-end NTN network including:

  * A moving LEO satellite constellation — satellites pass overhead and
    the UE hands over to the next satellite when the serving one drops
    below SAT_HANDOVER_ELEVATION_DEG.
  * RT-informed channel model — the packet error rate on the service
    link is derived from the mean path gain computed by Sionna RT in
    rt_sim.py, rather than from a purely analytic FSPL formula.
  * Multi-protocol comparison — the function run_ns3_both_topologies()
    runs the full simulation once per entry in config.PROTOCOLS × 2
    topologies (direct / indirect) and returns all results for plotting.
  * QUIC emulation — QUIC is emulated on top of NS-3 UDP with RFC 9000
    corrections applied analytically after FlowMonitor collection.
  * Full beam management handover — a 3-phase link interruption gap
    models beam failure detection → RACH → conditional handover
    per 3GPP TS 38.300 §10.1.2.3.
  * Mixed traffic profiles — each client is assigned a traffic profile
    (streaming / gaming / texting / voice / bulk) so that heterogeneous load is
    accurately modelled across all protocol comparison runs.
  * Per-second time-series — a PacketSink probe fires every
    TIMESERIES_BUCKET_S seconds to record true per-second throughput.
  * Jain's fairness index — computed across per-flow throughput values
    after FlowMonitor collection.

Topology
--------

  Direct (3 nodes):
    Phone (UE)
       │  5G-NR NTN service link  (550 km LEO, RT-calibrated PER, PHONE_EIRP)
       ▼
    Satellite [moving]     ← handover when elevation < SAT_HANDOVER_ELEVATION_DEG
       │  Direct link  (SAT_SERVER_DATARATE, 1 ms, lossless)
       ▼
    Internet Server

Beam management handover (3GPP TS 38.300 §10.1.2.3)
----------------------------------------------------
Each handover consists of three sequential phases:

  Phase 1 — Beam Failure Detection (BEAM_FAILURE_DETECTION_MS):
    The UE detects that the serving beam has dropped below the handover
    threshold.  During this window the link is unreliable (PER → 1.0).

  Phase 2 — Random Access (RANDOM_ACCESS_DELAY_MS):
    The UE performs PRACH on the target satellite beam.

  Phase 3 — Conditional Handover Preparation (CONDITIONAL_HO_PREP_MS):
    RRC reconfiguration and path switch complete.  The link recovers
    with the new satellite's PER.

Total interruption gap is drawn uniformly from
[HANDOVER_INTERRUPTION_MS_MIN, HANDOVER_INTERRUPTION_MS_MAX].
The gap applies to the NTN hop in both direct and indirect topologies.

QUIC emulation (RFC 9000 / RFC 9002)
-------------------------------------
QUIC has no native NS-3 module; it is emulated over UDP with mechanism-by-
mechanism post-FlowMonitor corrections derived from first principles:

  1. 1-RTT handshake saving:    QUIC needs 1 RTT vs TCP's 2 RTT to start
                                 sending data.  Extra throughput credit =
                                 (mean_RTT_s × link_rate_bps) × 8 / active_s.

  2. PTO vs RTO (no cwnd→1):   QUIC's Probe Timeout fires probe packets
                                 WITHOUT collapsing cwnd to 1 MSS.  At
                                 high PER, TCP RTO events collapse cwnd
                                 repeatedly; QUIC recovers faster.
                                 Credit = cwnd_floor × MSS × rto_rate × 8 / 1e3

  3. Unlimited ACK ranges:      QUIC ACK frames support unlimited gap/range
                                 pairs vs TCP SACK's 3 blocks.  At PER > 0.5
                                 the loss map requires many ACK round trips
                                 for TCP; QUIC resolves it in one.  Applied
                                 at high-PER slots only.

  4. Post-handover recovery:    QUIC PATH_CHALLENGE/RESPONSE costs 1 RTT and
                                 preserves ssthresh; TCP breaks the connection
                                 entirely.  The post-handover throughput dip
                                 is reduced vs TCP BBR.  Beam interruption gap
                                 is shared by both protocols; only recovery
                                 RTTs differ.

  5. HoL blocking:              NOT applied — zero benefit for single bulk
                                 stream (only matters for multi-stream HTTP/3).

The QUIC baseline is TCP BBR (BBR is the congestion controller used by
real QUIC implementations such as Chromium/quiche and lsquic for satellite
links).  Corrections are additive on top of the BBR aggregate result.

Dependencies
------------
  NS-3 with Python bindings (cppyy-based, 3.37+).
  See README.md for build instructions.
"""

import math
import random as _random_mod
import sys
import os
import numpy as np

# Ensure the NS-3 cppyy Python bindings directory is on sys.path.
# The bindings are built in-tree at ns-3-dev/build/bindings/python/
# and are not installed into the venv's site-packages.
_NS3_BINDINGS = os.path.expanduser("~/ns-3-dev/build/bindings/python")
if _NS3_BINDINGS not in sys.path:
    sys.path.insert(0, _NS3_BINDINGS)

from ns import ns   # NS-3 cppyy-based Python bindings

# ---------------------------------------------------------------------------
# C++ helper: wrap a no-arg Python callable as an NS-3 EventImpl*
# This is required because the cppyy bindings cannot dispatch plain Python
# functions to ns3::Simulator::Schedule directly.  The pattern comes from
# the official NS-3 Python example tcp-bbr-example.py.
# ---------------------------------------------------------------------------
ns.cppyy.cppdef(r"""
#include "ns3/simulator.h"
using namespace ns3;
EventImpl* pythonMakeEvent(void (*f)()) { return MakeEvent(f); }
""")

from config import (
    CARRIER_FREQ_HZ,
    SAT_HEIGHT_M,
    SAT_ORBITAL_VELOCITY_MS,
    SAT_HANDOVER_ELEVATION_DEG,
    SIM_DURATION_S,
    APP_DATA_RATE,
    PACKET_SIZE_BYTES,
    NUM_PARALLEL_FLOWS,
    TCP_SNDRCV_BUF_BYTES,
    TCP_SACK_ENABLED,
    TCP_TIMESTAMPS,
    PROTOCOLS,
    PHONE_EIRP_DBM,
    SAT_RX_ANTENNA_GAIN_DB,
    NOISE_FLOOR_DBM,
    SNR_THRESH_DB,
    SIGMOID_SLOPE,
    RT_GAIN_P10_BLEND,
    ATMO_GASEOUS_DB,
    RAIN_RATE_MM_H,
    RAIN_AVAILABILITY_PCT,
    ATMO_SCINTILLATION_DB,
    # Multi-client topology
    NUM_STATIONARY_CLIENTS,
    NUM_MOVING_CLIENTS,
    CLIENT_AREA_RADIUS_M,
    NUM_PEDESTRIAN_MOVING_CLIENTS,
    NUM_VEHICULAR_MOVING_CLIENTS,
    PEDESTRIAN_SPEED_MIN_MS,
    PEDESTRIAN_SPEED_MAX_MS,
    VEHICULAR_SPEED_MIN_MS,
    VEHICULAR_SPEED_MAX_MS,
    DATA_VOLUME_MB,
    SAT_SERVER_DATARATE,
    SERVICE_LINK_RATE_MBPS,
    BACKHAUL_DELAY_MS,
    # Beam management
    HANDOVER_INTERRUPTION_MS_MIN,
    HANDOVER_INTERRUPTION_MS_MAX,
    BEAM_FAILURE_DETECTION_MS,
    RANDOM_ACCESS_DELAY_MS,
    CONDITIONAL_HO_PREP_MS,
    HANDOVER_PHASE_JITTER_MS,
    # Traffic profiles
    TRAFFIC_PROFILES,
    CLIENT_PROFILES,
    TRANSPORT_COMPARE_MODE,
    MATCHED_PROFILE_DATA_RATES,
    QUIC_IS_ANALYTICAL,
    # Time-series
    TIMESERIES_BUCKET_S,
    # Burst-fading model
    FADE_MEAN_DURATION_MS,
    # Reproducibility
    SIM_SEED,
    SIM_RUN,
    PY_RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# NTN block-fading / slow-fading note (C1 thesis caveat)
#
# A naive fix would apply  PER = 1 − (1−BLER)^N  where N = ⌈1400 B / 192 B⌉ = 8
# codewords per IP packet, on the grounds that a packet error requires any one
# codeword to fail.  That formula is only valid when codeword errors are
# INDEPENDENT, i.e. the channel changes between codewords (fast fading).
#
# NTN is the opposite extreme — slow / block fading:
#   - Coherence time ≫ packet duration (≈ 4 ms for 8 × 0.5 ms slots).
#   - Over 4 ms at 550 km the satellite moves ~30 m → negligible channel change.
#   - All 8 codewords in one packet therefore see the SAME instantaneous SNR
#     and fail or succeed together.
#
# In the block-fading model:  PER_packet ≈ BLER_codeword
# which is exactly what the sigmoid returns.  Using BLER directly as PER is
# therefore the CORRECT assumption for NTN, not an approximation to be fixed.
#
# The real thesis caveat is the absence of HARQ-IR: incremental redundancy
# would reduce the effective per-transmission BLER by ~3 dB but is not modelled.
# Reference: 3GPP TR 38.821 §6.1.4 (HARQ in NTN), TS 38.212 §5.4.2 (IR-HARQ).

# =============================================================================
# Application-layer KPI helpers (ITU-T G.107 / empirical mappings)
# =============================================================================

def _g107_mos(latency_ms: float, loss_pct: float,
              codec_ie: float = 0.0, codec_bpl: float = 25.1) -> float:
    """
    ITU-T G.107 E-model simplified R-factor → MOS mapping.

    Default codec equipment impairments (I_e, B_pl) correspond to G.711
    PCM with packet-loss robustness factor B_pl = 25.1.

    Returns MOS clamped to [1.0, 4.5] (G.107 anchors MOS_max ≈ 4.5 for
    G.711-quality speech).
    """
    R_0 = 93.2
    T   = max(float(latency_ms), 0.0)
    # Delay impairment I_d (Eq. 7-11a/b of G.107):
    Id = 0.024 * T
    if T > 177.3:
        Id += 0.11 * (T - 177.3)
    # Equipment impairment under random packet loss:
    Ppl = max(min(float(loss_pct), 99.9), 0.0)
    Ie_eff = codec_ie + (95.0 - codec_ie) * Ppl / (Ppl + codec_bpl)

    R = R_0 - Id - Ie_eff
    if R < 0.0:
        mos = 1.0
    elif R > 100.0:
        mos = 4.5
    else:
        mos = 1.0 + 0.035 * R + 7e-6 * R * (R - 60.0) * (100.0 - R)
    return float(max(1.0, min(mos, 4.5)))


def _video_psnr_db(mean_bitrate_kbps: float) -> float:
    """
    Empirical H.264 CBR bitrate → PSNR mapping for 480p–720p video.

    PSNR ≈ 10 + 10·log10(bitrate_kbps), saturating at 48 dB for very high
    bitrates (visual-quality ceiling).
    """
    if mean_bitrate_kbps <= 0.0:
        return 0.0
    return float(min(10.0 + 10.0 * math.log10(mean_bitrate_kbps), 48.0))


# =============================================================================
# Geometry and link-budget helpers
# =============================================================================

def _slant_range_m(height_m: float, elev_deg: float) -> float:
    """
    One-way slant range from UE to satellite [m] using spherical Earth
    geometry.

    Parameters
    ----------
    height_m : float  Satellite orbital altitude [m].
    elev_deg : float  UE-to-satellite elevation angle [degrees].
    """
    RE = 6_371_000.0
    e  = math.radians(elev_deg)
    return (math.sqrt((RE + height_m) ** 2 - (RE * math.cos(e)) ** 2)
            - RE * math.sin(e))


def _one_way_delay_ms(height_m: float, elev_deg: float) -> float:
    """One-way propagation delay [ms] along the slant range."""
    return _slant_range_m(height_m, elev_deg) / 3e8 * 1e3


def _fspl_db(height_m: float, elev_deg: float,
             fc_hz: float = CARRIER_FREQ_HZ) -> float:
    """
    Free-space path loss [dB] at carrier frequency ``fc_hz``.

    FSPL = 20·log10(4π·d·f / c)
    """
    d = _slant_range_m(height_m, elev_deg)
    return 20.0 * math.log10(4.0 * math.pi * d * fc_hz / 3e8)


def _itu_p618_rain_fade_db(elev_deg: float,
                           freq_hz: float = CARRIER_FREQ_HZ,
                           rain_rate_mm_h: float = None,
                           availability_pct: float = None) -> float:
    """
    ITU-R P.618-13 §2.2.1 rain attenuation [dB] on a LEO slant path.

    Simplified model for availability ≥ 99%:

      γ_R = k · R^α                                [dB/km, specific attenuation]
      L_s = (h_R − h_s) / sin(elev)                [km, slant-path length]
      r   = 1 / (1 + L_s · cos(elev) / L_0)        [horizontal reduction factor]
      A_0.01 = γ_R · L_s · r                       [dB at 0.01% exceedance]
      A_p  = A_0.01 · (p/0.01)^(-(0.655 + 0.033·ln(p) − 0.045·ln(A_0.01)))

    where p = (100 − availability_pct). At 2 GHz the (k, α) coefficients
    (ITU-R P.838-3) are small (k ≈ 2.7e-5, α ≈ 1.22), so S-band rain fade
    is typically well under 1 dB even at moderate rain rates.
    """
    if rain_rate_mm_h is None:
        rain_rate_mm_h = RAIN_RATE_MM_H
    if availability_pct is None:
        availability_pct = RAIN_AVAILABILITY_PCT

    if elev_deg <= 0.0 or rain_rate_mm_h <= 0.0:
        return 0.0

    f_ghz = freq_hz / 1e9
    # ITU-R P.838-3 (k, α) coefficients for vertical polarization at 2 GHz.
    # Log-log interpolation between the 1 GHz and 4 GHz tabulated points.
    k = 10.0 ** (-4.88 + 1.93 * math.log10(f_ghz))
    alpha = 0.94 + 0.16 * math.log10(f_ghz)
    gamma_R = k * (rain_rate_mm_h ** alpha)

    h_R = 3.0    # km, 0°C isotherm height (ITU-R P.839-4 temperate latitude)
    h_s = 0.03   # km, station height (street level)
    e_rad = math.radians(max(elev_deg, 5.0))
    L_s = (h_R - h_s) / math.sin(e_rad)

    L_0 = 35.0 * math.exp(-0.015 * rain_rate_mm_h)
    r = 1.0 / (1.0 + (L_s * math.cos(e_rad) / L_0))

    A_001 = gamma_R * L_s * r

    p = max(100.0 - availability_pct, 1e-3)
    if p >= 0.01 and A_001 > 0.0:
        exp_factor = -(0.655 + 0.033 * math.log(p) - 0.045 * math.log(A_001))
        A_p = A_001 * (p / 0.01) ** exp_factor
    else:
        A_p = A_001

    return float(max(A_p, 0.0))


def _itu_p618_scintillation_db(elev_deg: float) -> float:
    """
    ITU-R P.618-13 §2.4.1 tropospheric scintillation fade [dB].

    Proportional to 1/sin(elev)^1.2; returns the configured σ at zenith,
    scaled to the given elevation. Never dominates at S-band but included
    as a completeness check against ITU-R P.618.
    """
    e_rad = math.radians(max(elev_deg, 5.0))
    return float(ATMO_SCINTILLATION_DB / (math.sin(e_rad) ** 1.2))


def _rt_calibrated_per(fspl_db: float, rt_mean_gain_db: float,
                        rt_gain_p10_db: float = None,
                        rt_ref_gain_db: float = None,
                        snr_thresh_db: float = None,
                        sigmoid_slope: float = None,
                        elev_deg: float = None) -> float:
    """
    Estimate packet error rate using a sigmoid link-budget model
    calibrated by the RT-derived mean path gain.

    Link budget (direct Phone→Satellite uplink):
        SNR [dB] = PHONE_EIRP − FSPL_550km + urban_correction
                   + SAT_RX_GAIN − NOISE_FLOOR

    urban_correction normalises the RT gain against the best satellite so
    that the reference contributes 0 dB and lower-elevation satellites are
    penalised by their relative RT gain deficit.

    Parameters
    ----------
    fspl_db         : float  Free-space path loss over real 550 km slant [dB].
    rt_mean_gain_db : float  Mean |h| gain from Sionna RT for this satellite [dB].
    rt_gain_p10_db  : float  10th-percentile gain (optional urban penalty).
    rt_ref_gain_db  : float  RT gain of the reference (best) satellite [dB].
                             When None, no urban correction is applied.
    snr_thresh_db   : float  Sigmoid threshold [dB] (default: SNR_THRESH_DB).
    sigmoid_slope   : float  Sigmoid slope [1/dB] (default: SIGMOID_SLOPE).

    Returns
    -------
    float  Packet error rate in [0, 1).
    """
    if snr_thresh_db is None:
        snr_thresh_db = SNR_THRESH_DB
    if sigmoid_slope is None:
        sigmoid_slope = SIGMOID_SLOPE

    # Urban multipath correction: relative shadow-fading offset from the best
    # satellite, using proxy-FSPL-normalised gains (normalized_gain_db from
    # channel_stats).  Normalisation removes the geometric bias introduced by
    # proxy transmitters at different scene distances, leaving only the true
    # building-shadow / multipath contribution.
    #
    # The correction is capped at ±12 dB (3σ of 3GPP TR 38.811 Urban NTN LOS
    # shadow fading σ = 4 dB) to prevent rare scene outliers from producing
    # unrealistically high PER.
    #
    # If the normalised gain is NaN (no valid RT paths for this satellite),
    # fall back to a −10 dB conservative shadow-fading penalty.
    _URBAN_CORRECTION_CAP_DB = 12.0   # 3σ LOS per 3GPP TR 38.811 Table 6.7.2-1

    no_paths = math.isnan(rt_mean_gain_db) if isinstance(rt_mean_gain_db, float) else False
    gain_for_budget_db = rt_mean_gain_db
    if rt_gain_p10_db is not None and not math.isnan(rt_gain_p10_db):
        gain_for_budget_db = (
            (1.0 - RT_GAIN_P10_BLEND) * rt_mean_gain_db
            + RT_GAIN_P10_BLEND * rt_gain_p10_db
        )

    if no_paths or math.isnan(gain_for_budget_db):
        urban_correction_db = -10.0    # No paths → deep shadow penalty
    elif rt_ref_gain_db is not None and not math.isnan(rt_ref_gain_db):
        raw_correction = gain_for_budget_db - rt_ref_gain_db
        urban_correction_db = float(np.clip(raw_correction,
                                            -_URBAN_CORRECTION_CAP_DB,
                                            _URBAN_CORRECTION_CAP_DB))
    else:
        urban_correction_db = 0.0

    # Full uplink budget:
    #   SNR = TX_EIRP − FSPL − atm_gaseous − rain_fade − scintillation
    #         + urban_correction + SAT_RX_GAIN − NOISE_FLOOR
    # SAT_RX_ANTENNA_GAIN_DB models the satellite's phased-array receive
    # aperture (34 dBi at 2 GHz for a modern LEO NTN spot beam).
    # Atmospheric terms (ITU-R P.618/P.676) are applied only when elev_deg
    # is provided, to preserve backward compatibility with callers that
    # cannot supply geometry (e.g. per-flow plot helpers).
    atm_loss_db = 0.0
    if elev_deg is not None:
        atm_loss_db = (ATMO_GASEOUS_DB
                       + _itu_p618_rain_fade_db(elev_deg)
                       + _itu_p618_scintillation_db(elev_deg))

    snr_db = (PHONE_EIRP_DBM - fspl_db - atm_loss_db + urban_correction_db
              + SAT_RX_ANTENNA_GAIN_DB - NOISE_FLOOR_DBM)
    # Block-fading (NTN slow-fading): PER_packet ≈ BLER_codeword.
    # All codewords in one packet share the same channel state → use BLER directly.
    per    = 1.0 / (1.0 + math.exp(sigmoid_slope * (snr_db - snr_thresh_db)))
    return float(np.clip(per, 0.0, 0.99))


# =============================================================================
# Handover schedule computation
# =============================================================================

def _interp_channel_stats(sat_id: int, t_mid: float,
                          snapshots: list, times: list) -> dict | None:
    """
    Linearly interpolate per-satellite channel stats at simulation time t_mid.

    Returns None when snapshots is None or the sat_id is absent in a snapshot
    (caller falls back to the static t=0 stats in that case).

    Only the gain and DS fields are interpolated; non-numeric fields
    (sat_id, los_exists, sat_x_m/y_m) are taken from the earlier snapshot.
    """
    if not snapshots or len(snapshots) < 2:
        return None
    t = max(times[0], min(times[-1], t_mid))
    # Find bracket [i, i+1]
    idx = 0
    for i in range(len(times) - 1):
        if times[i] <= t <= times[i + 1]:
            idx = i
            break
    alpha = (t - times[idx]) / max(times[idx + 1] - times[idx], 1e-9)
    by_id_a = {s["sat_id"]: s for s in snapshots[idx]}
    by_id_b = {s["sat_id"]: s for s in snapshots[idx + 1]}
    a = by_id_a.get(sat_id)
    b = by_id_b.get(sat_id)
    if a is None or b is None:
        return None
    def _getv(d, k, default=float("nan")):
        return d[k] if k in d else default
    def _safe_lerp(va, vb):
        # NaN guard: when num_paths==0 a gain field is NaN; prefer the finite side.
        if math.isnan(va): return vb
        if math.isnan(vb): return va
        return va * (1.0 - alpha) + vb * alpha
    return {
        **a,
        "elevation_deg":      _safe_lerp(float(_getv(a, "elevation_deg", 0.0)),
                                          float(_getv(b, "elevation_deg", 0.0))),
        "normalized_gain_db": _safe_lerp(float(_getv(a, "normalized_gain_db")),
                                          float(_getv(b, "normalized_gain_db"))),
        "normalized_p10_db":  _safe_lerp(float(_getv(a, "normalized_p10_db")),
                                          float(_getv(b, "normalized_p10_db"))),
        "delay_spread_ns":    _safe_lerp(float(_getv(a, "delay_spread_ns", 0.0)),
                                          float(_getv(b, "delay_spread_ns", 0.0))),
        "k_factor_db":        _safe_lerp(float(_getv(a, "k_factor_db")),
                                          float(_getv(b, "k_factor_db"))),
    }


def _compute_handover_schedule(channel_stats: list,
                                snr_thresh_db: float = None,
                                sigmoid_slope: float = None,
                                rng_seed: int = 42,
                                channel_stats_snapshots: list = None,
                                snapshot_times: list = None) -> list:
    """
    Determine the sequence of (satellite_id, start_time, end_time,
    per, delay_ms, interruption_ms) intervals for the simulation duration.

    The algorithm:
    1. Sort satellites by decreasing elevation (highest = first to serve).
    2. Assign time slices weighted by slant_range / (v_orb × sin(elev)):
       satellites near the horizon are visible for longer in a real orbital
       pass, so they receive proportionally more simulation time.
    3. Satellites below SAT_HANDOVER_ELEVATION_DEG are skipped (link drop).
    4. Each handover transition is assigned a beam interruption gap drawn
       uniformly from [HANDOVER_INTERRUPTION_MS_MIN, HANDOVER_INTERRUPTION_MS_MAX]
       per 3GPP TS 38.300 §10.1.2.3.

    Parameters
    ----------
    channel_stats : list[dict]
        RT channel statistics, one dict per satellite (output of
        rt_sim.run_ray_tracing()).
    snr_thresh_db : float, optional
        Sigmoid threshold [dB].  Defaults to SNR_THRESH_DB when None.
        Pass the value fitted to the Sionna LDPC BER curve.
    sigmoid_slope : float, optional
        Sigmoid slope [1/dB].  Defaults to SIGMOID_SLOPE when None.
        Pass the value fitted to the Sionna LDPC BER curve.
    rng_seed : int
        Seed for the interruption-gap RNG (reproducibility).

    Returns
    -------
    list of dicts with keys:
        sat_id           int     Satellite index.
        t_start          float   Start of service window [s].
        t_end            float   End of service window [s].
        elev_deg         float   Elevation angle [degrees].
        per              float   Service-link PER (RT-calibrated).
        delay_ms         float   Service-link one-way propagation delay [ms].
        interruption_ms  float   Beam gap before this slot starts [ms].
                                 Zero for the first slot (no incoming HO).
    """
    rng = _random_mod.Random(rng_seed)

    def _phase_duration_ms(base_ms: float) -> float:
        j = HANDOVER_PHASE_JITTER_MS
        if j <= 0.0:
            return max(base_ms, 0.0)
        return max(base_ms + rng.uniform(-j, j), 0.0)

    # Sort by elevation, descending (highest elevation served first)
    sorted_stats = sorted(channel_stats, key=lambda s: s["elevation_deg"],
                          reverse=True)

    # Keep only satellites above the handover threshold
    visible = [s for s in sorted_stats
               if s["elevation_deg"] >= SAT_HANDOVER_ELEVATION_DEG]

    if not visible:
        # Fallback: use all satellites even if below threshold
        visible = sorted_stats

    # Slot duration weighting: satellites near the horizon spend more time
    # above the handover threshold during a real orbital pass.
    #
    # The angular rate of elevation change for a satellite on a zenith-pass
    # trajectory is approximately:
    #   d(elevation)/dt ≈ v_orb × sin(elevation) / slant_range(elevation)
    # A satellite at LOW elevation has a SMALL d(elev)/dt → it stays near that
    # elevation for LONGER.  The dwell time per unit elevation is therefore
    # proportional to:
    #   weight ∝ 1 / |d(elev)/dt| ∝ slant_range / (v_orb × sin(elevation))
    #
    # Reference: orbital mechanics (Wertz, "Space Mission Engineering", §9.1;
    #            3GPP TR 38.821 §6.1 orbit geometry).
    elev_rads   = [math.radians(max(s["elevation_deg"], 1.0)) for s in visible]
    raw_weights = [
        _slant_range_m(SAT_HEIGHT_M, max(s["elevation_deg"], 1.0))
        / (SAT_ORBITAL_VELOCITY_MS * math.sin(e))
        for s, e in zip(visible, elev_rads)
    ]
    total_w        = sum(raw_weights)
    slot_durations = [w / total_w * SIM_DURATION_S for w in raw_weights]

    schedule = []

    # Reference gain: proxy-FSPL-normalised gain of the best visible satellite.
    # Using normalised gains (mean_path_gain_db + proxy_fspl_db) removes the
    # geometric bias introduced by proxy TXs at different scene distances, so
    # the urban correction purely reflects shadow-fading differences.
    def _norm_gain(stat):
        v = stat.get("normalized_gain_db", float("nan"))
        return v if not math.isnan(v) else float("-inf")

    valid_visible = [s for s in visible if not math.isnan(_norm_gain(s))]
    ref_gain_db = max((_norm_gain(s) for s in valid_visible), default=None)

    t_cursor = 0.0
    for i, stat in enumerate(visible):
        slot_s   = slot_durations[i]
        t_start  = t_cursor
        t_end    = t_cursor + slot_s
        t_cursor = t_end
        elev_deg = stat["elevation_deg"]

        # R2 fix: if multi-snapshot stats are available, interpolate the
        # channel parameters at this slot's midpoint time.
        t_mid = 0.5 * (t_start + t_end)
        interp = _interp_channel_stats(
            stat["sat_id"], t_mid, channel_stats_snapshots, snapshot_times or []
        )
        eff_stat = interp if interp is not None else stat
        elev_eff = float(eff_stat.get("elevation_deg", elev_deg))

        fspl   = _fspl_db(SAT_HEIGHT_M, max(elev_eff, 1.0))
        per    = _rt_calibrated_per(fspl,
                                    eff_stat.get("normalized_gain_db", float("nan")),
                                    eff_stat.get("normalized_p10_db"),
                                    ref_gain_db,
                                    snr_thresh_db, sigmoid_slope,
                                    elev_deg=max(elev_eff, 1.0))
        delay  = _one_way_delay_ms(SAT_HEIGHT_M, max(elev_eff, 1.0))

        # Capture atmospheric / rain / scintillation components for reporting
        rain_db  = _itu_p618_rain_fade_db(max(elev_eff, 1.0))
        scint_db = _itu_p618_scintillation_db(max(elev_eff, 1.0))

        # Beam interruption: zero for first slot. For subsequent slots,
        # model BFD + RACH + CHO phase durations with small jitter.
        if i == 0:
            gap_ms = 0.0
            bfd_ms = 0.0
            rach_ms = 0.0
            cho_ms = 0.0
        else:
            bfd_ms = _phase_duration_ms(BEAM_FAILURE_DETECTION_MS)
            rach_ms = _phase_duration_ms(RANDOM_ACCESS_DELAY_MS)
            cho_ms = _phase_duration_ms(CONDITIONAL_HO_PREP_MS)
            gap_ms = bfd_ms + rach_ms + cho_ms
            gap_ms = float(np.clip(gap_ms,
                                   HANDOVER_INTERRUPTION_MS_MIN,
                                   HANDOVER_INTERRUPTION_MS_MAX))

        schedule.append(dict(
            sat_id         = stat["sat_id"],
            t_start        = round(t_start, 3),
            t_end          = round(t_end, 3),
            elev_deg       = round(elev_eff, 2),
            per            = round(per, 4),
            delay_ms       = round(delay, 2),
            interruption_ms = round(gap_ms, 1),
            bfd_ms         = round(bfd_ms, 1),
            rach_ms        = round(rach_ms, 1),
            cho_ms         = round(cho_ms, 1),
            rain_fade_db   = round(rain_db, 3),
            scint_db       = round(scint_db, 3),
            atm_gaseous_db = round(ATMO_GASEOUS_DB, 2),
        ))

    return schedule


# =============================================================================
# NS-3 topology helpers
# =============================================================================

def _nc2(a, b):
    """NodeContainer from two nodes (cppyy lacks the 2-arg constructor)."""
    nc = ns.NodeContainer()
    nc.Add(a)
    nc.Add(b)
    return nc


def _configure_tcp(protocol_cfg: dict) -> None:
    """
    Apply global NS-3 TCP stack configuration.

    Called once before creating any sockets so that the settings are
    visible to all TCP connections in the simulation.

    Parameters
    ----------
    protocol_cfg : dict  Protocol configuration entry from config.PROTOCOLS
                         augmented with TCP_* keys from config.py.
    """
    variant  = protocol_cfg.get("tcp_variant", "Cubic") or "Cubic"
    snd_buf  = protocol_cfg.get("tcp_snd_buf",  TCP_SNDRCV_BUF_BYTES)
    rcv_buf  = protocol_cfg.get("tcp_rcv_buf",  TCP_SNDRCV_BUF_BYTES)
    sack     = protocol_cfg.get("tcp_sack",     TCP_SACK_ENABLED)
    ts       = protocol_cfg.get("tcp_timestamps", TCP_TIMESTAMPS)

    # Congestion control algorithm
    ns.Config.SetDefault(
        "ns3::TcpL4Protocol::SocketType",
        ns.StringValue(f"ns3::Tcp{variant}"))

    # Socket buffer sizes (critical for NTN high-BDP paths)
    ns.Config.SetDefault(
        "ns3::TcpSocket::SndBufSize",
        ns.UintegerValue(snd_buf))
    ns.Config.SetDefault(
        "ns3::TcpSocket::RcvBufSize",
        ns.UintegerValue(rcv_buf))

    # SACK
    ns.Config.SetDefault(
        "ns3::TcpSocketBase::Sack",
        ns.BooleanValue(bool(sack)))

    # Timestamps
    ns.Config.SetDefault(
        "ns3::TcpSocketBase::Timestamp",
        ns.BooleanValue(bool(ts)))


# =============================================================================
# QUIC post-FlowMonitor corrections
# =============================================================================

def _apply_quic_corrections(bbr_result: dict, schedule: list,
                             link_rate_bps: float = None) -> dict:
    """
    Apply RFC 9000/9002 mechanism corrections on top of a TCP BBR
    FlowMonitor result to produce a QUIC emulation result.

    QUIC is emulated over UDP in NS-3.  This function takes the
    aggregate BBR result (the best TCP baseline for satellite) and
    computes additive throughput/latency corrections for each QUIC
    mechanism that has a measurable effect on a single bulk flow over
    an NTN link.

    Corrections applied
    -------------------
    1. 1-RTT handshake saving
       QUIC completes the handshake in 1 RTT (vs TCP+TLS 1.3 = 2 RTTs),
       saving one RTT worth of bandwidth that TCP wastes in the slow-start
       phase.
       Δtput += (mean_RTT_s × link_rate_bps) / effective_active_s   [kbps]

    2. PTO vs RTO (no cwnd→1 collapse)
       QUIC's Probe Timeout fires 1–2 probe packets without resetting cwnd
       to 1 MSS.  At slots where PER > fast-retransmit threshold (~0.01),
       TCP RTO events repeatedly collapse cwnd to 1; QUIC does not.
       Δtput += cwnd_floor_pkts × MSS × rto_rate × 8 / 1e3   [kbps]
       where rto_rate = estimated RTO events per second based on PER and RTT.
       cwnd_floor_pkts = 2 per RFC 9002 §6.2.4 (minimum 2 packets in flight
       during PTO probing).

    3. Unlimited ACK ranges
       At PER > 0.5, TCP SACK (3 block limit) cannot describe the full loss
       map in one ACK, requiring multiple ACK round trips.  QUIC ACK frames
       support unlimited gap/range pairs → 1 ACK suffices.
       Applied as a latency reduction at high-PER slots.
       Source: IETF draft-kuhn-quic-4-sat Table 1 (Kuhn et al., 2020).

    4. Post-handover throughput dip reduction
       QUIC connection migration (PATH_CHALLENGE/PATH_RESPONSE) costs 1 RTT
       after the beam gap ends and preserves ssthresh.  TCP requires a full
       slow-start (~9 RTTs) after the gap.  Both protocols are subject to
       the same RF blackout gap; only recovery RTTs differ.
       Source: RFC 9000 §9.3.

    5. Loss reduction via PTO vs RTO
       QUIC PTO probes resolve isolated losses without RTO; effective loss
       fraction decays exponentially with RTT relative to the reference RTT.
        Reduction factor = exp(-0.8 × mean_RTT_s / REF_ONE_WAY_DELAY_S) × PER_ratio
        where REF_ONE_WAY_DELAY_S = 0.035 s (35 ms one-way reference delay,
        half of the 70 ms median LEO RTT reported in Sander et al. IMC 2022 and
        IETF draft-kuhn-quic-4-sat Table 1; used as the normalisation denominator).
       PER_ratio = mean(slot PER) over the schedule.

    6. HoL blocking elimination
       NOT applied — zero benefit for a single bulk stream.  HoL blocking
       elimination only matters for multi-stream / HTTP/3 workloads.

    Beam interruption gap
    ---------------------
    The total beam blackout time across all handovers is subtracted from
    the effective active simulation duration:
        total_gap_s = sum(slot["interruption_ms"] for all slots) / 1000
        effective_active_s = active_s - total_gap_s
    Both QUIC and TCP experience the same RF blackout; QUIC recovers in
    1 RTT after the gap ends vs TCP's ~9 RTTs.

    Parameters
    ----------
    bbr_result    : dict   FlowMonitor result dict from a TCP BBR run.
    schedule      : list   Handover schedule (list of slot dicts with per, delay_ms,
                           interruption_ms).
    link_rate_bps : float  Actual service-link rate [bps].  When None, uses
                           10 Mbps (the service link data rate in run_ns3).

    Returns
    -------
    dict  QUIC result dict with corrected throughput_kbps, mean_delay_ms,
          loss_pct, and updated label/protocol fields.
    """
    import copy
    result = copy.deepcopy(bbr_result)
    result["protocol"] = "quic"
    result["label"]    = "QUIC (analytical)" if QUIC_IS_ANALYTICAL else "QUIC"
    result["is_analytical"] = bool(QUIC_IS_ANALYTICAL)

    # If BBR got zero throughput the link is effectively dead (e.g. direct
    # topology with phone EIRP is too weak).  QUIC cannot overcome a
    # broken link — return zero throughput with a nominal 1 ms latency.
    if bbr_result["throughput_kbps"] == 0.0 and bbr_result["mean_delay_ms"] == 0.0:
        result["throughput_kbps"] = 0.0
        result["mean_delay_ms"]   = 1.0
        result["loss_pct"]        = 0.0
        print(f"\n  QUIC corrections (from BBR baseline):")
        print(f"    BBR baseline = 0 kbps (link failed) — QUIC result = 0 kbps")
        return result

    if link_rate_bps is None:
        link_rate_bps = SERVICE_LINK_RATE_MBPS * 1e6

    # RFC 9000 §9.3 post-handover recovery RTT counts
    TCP_RECOVERY_RTTS  = 9    # TCP slow-start + SYN from scratch
    QUIC_RECOVERY_RTTS = 1    # QUIC PATH_CHALLENGE/RESPONSE

    # Reference one-way delay for loss-reduction formula (Sander et al.
    # IMC 2022 — median LEO RTT = 70 ms → one-way = 35 ms; also consistent
    # with IETF draft-kuhn-quic-4-sat Table 1).
    REF_ONE_WAY_DELAY_S = 0.035   # 35 ms one-way

    mss_bytes  = PACKET_SIZE_BYTES
    active_s   = SIM_DURATION_S - 1.0
    mean_rtt_s = max(bbr_result["mean_delay_ms"] / 1e3 * 2.0, 0.001)
    handovers  = bbr_result.get("handovers", 0)

    # Total beam blackout across all handovers (shared by QUIC and TCP)
    total_gap_s = sum(s.get("interruption_ms", 0.0) for s in schedule) / 1000.0
    effective_active_s = max(active_s - total_gap_s, 1.0)

    delta_kbps = 0.0

    # ── 1. 1-RTT handshake saving ─────────────────────────────────────────────
    handshake_saving_kbps = (mean_rtt_s * link_rate_bps) / effective_active_s / 1e3
    delta_kbps += handshake_saving_kbps

    # ── 2. PTO vs RTO: no cwnd collapse ───────────────────────────────────────
    cwnd_floor_pkts = 2   # RFC 9002 §6.2.4: minimum 2 packets during PTO probing
    rto_credit_kbps = 0.0
    for slot in schedule:
        slot_duration_s = slot["t_end"] - slot["t_start"]
        slot_per        = slot["per"]
        if slot_per > 0.03:
            window_pkts = max(int(link_rate_bps * mean_rtt_s / (mss_bytes * 8)), 4)
            rto_rate    = slot_per / (mean_rtt_s * window_pkts + 1e-9)
            slot_credit = cwnd_floor_pkts * mss_bytes * rto_rate * 8 / 1e3
            rto_credit_kbps += slot_credit * (slot_duration_s / effective_active_s)
    delta_kbps += rto_credit_kbps

    # ── 3. Unlimited ACK ranges at high-PER slots ─────────────────────────────
    high_per_fraction = sum(
        (s["t_end"] - s["t_start"]) / effective_active_s
        for s in schedule if s["per"] > 0.5
    )
    latency_reduction_ms = bbr_result["mean_delay_ms"] * (2.0 / 3.0) * high_per_fraction

    # ── 4. Post-handover recovery ─────────────────────────────────────────────
    # Both QUIC and TCP stall for total_gap_s during beam blackout.
    # After the gap: QUIC needs 1 RTT, TCP needs ~9 RTTs to recover.
    if handovers > 0:
        handover_credit_kbps = (
            (TCP_RECOVERY_RTTS - QUIC_RECOVERY_RTTS)
            * mean_rtt_s * link_rate_bps / effective_active_s / 1e3
            * handovers
        )
        delta_kbps += handover_credit_kbps
    else:
        handover_credit_kbps = 0.0

    # ── 5. Loss reduction: PTO prevents cwnd collapse ─────────────────────────
    mean_per = (sum(s["per"] * (s["t_end"] - s["t_start"])
                    for s in schedule) / effective_active_s
                if schedule else 0.0)
    loss_reduction_factor = math.exp(-0.8 * mean_rtt_s / REF_ONE_WAY_DELAY_S) * max(mean_per, 1e-6)
    loss_reduction_factor = min(loss_reduction_factor, 0.99)   # cap at 99%

    # ── Apply corrections (capped by physical beam capacity) ──────────────────
    # QUIC is a transport-layer optimisation — it cannot exceed the shared
    # service-link rate, so cap the additive correction at link_rate_bps even
    # when TCP BBR is severely starved by congestion (otherwise the formula
    # can yield unphysical throughputs).
    raw_tput_kbps = bbr_result["throughput_kbps"] + delta_kbps
    link_ceiling_kbps = link_rate_bps / 1e3
    result["throughput_kbps"] = round(min(raw_tput_kbps, link_ceiling_kbps), 2)
    result["mean_delay_ms"]   = round(
        max(1.0, bbr_result["mean_delay_ms"] - latency_reduction_ms), 2)
    result["loss_pct"] = round(
        bbr_result["loss_pct"] * (1.0 - loss_reduction_factor), 3)

    print(f"\n  QUIC corrections (from BBR baseline):")
    print(f"    link_rate        : {link_rate_bps/1e6:.2f} Mbps")
    print(f"    beam gap (total) : {total_gap_s*1e3:.0f} ms  "
          f"(effective active = {effective_active_s:.1f} s)")
    print(f"    handshake saving : +{handshake_saving_kbps:.1f} kbps")
    print(f"    PTO vs RTO       : +{rto_credit_kbps:.1f} kbps")
    print(f"    post-H/O credit  : +{handover_credit_kbps:.1f} kbps  "
          f"(RFC 9000 §9.3: {TCP_RECOVERY_RTTS}-{QUIC_RECOVERY_RTTS} RTTs × {handovers} H/O)")
    print(f"    ACK range lat.   : -{latency_reduction_ms:.2f} ms  "
          f"(draft-kuhn-quic-4-sat Table 1)")
    print(f"    loss reduction   : ×{1.0-loss_reduction_factor:.3f}  "
          f"(exp(-0.8×RTT/{REF_ONE_WAY_DELAY_S*1e3:.0f}ms one-way) × PER={mean_per:.3f})")
    print(f"    total Δtput      : +{delta_kbps:.1f} kbps  →  {result['throughput_kbps']:.0f} kbps")
    print(f"    latency          : {bbr_result['mean_delay_ms']:.1f} ms → {result['mean_delay_ms']:.1f} ms")
    print(f"    loss             : {bbr_result['loss_pct']:.2f}% → {result['loss_pct']:.3f}%")

    return result


# =============================================================================
# Main NS-3 simulation run — multi-client ISL relay topology
# =============================================================================

def run_ns3(scenario: str, protocol_cfg: dict, channel_stats: list,
            snr_thresh_db: float = None,
            sigmoid_slope: float = None,
            sample_positions: bool = False,
            channel_stats_snapshots: list = None,
            snapshot_times: list = None) -> dict:
    """
    Run one full NS-3 5G-NTN simulation with:
      - NUM_STATIONARY_CLIENTS + NUM_MOVING_CLIENTS client phones
      - One satellite node (always the highest-elevation satellite)
      - Phone → Satellite → Server (no ISL or ground-station hops)
      - Moving clients use RandomWaypointMobilityModel
      - TCP BulkSend capped at DATA_VOLUME_MB
      - Mixed traffic profiles: streaming/gaming/texting/voice → UDP CBR,
        bulk → TCP
      - Full beam management: 3-phase link interruption gap per handover
        (simulated via error model updates on the phone→sat link)
      - Per-second throughput time-series via PacketSink probe
      - Jain's fairness index across per-flow throughputs

    Parameters
    ----------
    scenario : str
        Channel scenario label (e.g. ``"urban"``).  Used for display only.
    protocol_cfg : dict
        Protocol configuration dict from config.PROTOCOLS.
        For QUIC: runs as UDP internally; RFC 9000 corrections applied after.
    channel_stats : list[dict]
        RT channel statistics returned by rt_sim.run_ray_tracing().
    snr_thresh_db : float, optional
        Sigmoid threshold [dB] fitted to the Sionna LDPC BER curve.
    sigmoid_slope : float, optional
        Sigmoid slope [1/dB] fitted to the Sionna LDPC BER curve.

    Returns
    -------
    dict with keys:
        scenario, protocol, label, topology, elevation_deg, svc_delay_ms,
        svc_loss_pct, tx_packets, rx_packets, loss_pct,
        mean_delay_ms, jitter_ms, throughput_kbps, handovers, schedule,
        fairness_index, profile_stats, timeseries.
    """
    import collections
    import math as _math

    # Global reproducibility controls
    _random_mod.seed(PY_RANDOM_SEED)
    np.random.seed(SIM_SEED)
    ns.RngSeedManager.SetSeed(SIM_SEED)
    ns.RngSeedManager.SetRun(SIM_RUN)

    proto     = protocol_cfg["protocol"]
    label     = protocol_cfg.get("label", proto.upper())
    ns3_proto = "udp" if proto == "quic" else proto

    num_clients = NUM_STATIONARY_CLIENTS + NUM_MOVING_CLIENTS

    print(f"\n[NS-3]  {scenario.upper()}  topology=direct  "
          f"protocol={label}  clients={num_clients}")

    # ── TCP global config ─────────────────────────────────────────────────────
    if ns3_proto == "tcp":
        _configure_tcp(protocol_cfg)

    # ── Handover schedule ─────────────────────────────────────────────────────
    schedule = _compute_handover_schedule(
        channel_stats,
        snr_thresh_db=snr_thresh_db,
        sigmoid_slope=sigmoid_slope,
        channel_stats_snapshots=channel_stats_snapshots,
        snapshot_times=snapshot_times,
    )
    print(f"  Handover schedule (EIRP={PHONE_EIRP_DBM:.0f} dBm): {len(schedule)} slot(s)")
    for slot in schedule:
        gap_str = f"  gap={slot['interruption_ms']:.0f}ms" if slot["interruption_ms"] > 0 else ""
        atm_total = (slot.get("atm_gaseous_db", 0.0)
                     + slot.get("rain_fade_db", 0.0)
                     + slot.get("scint_db", 0.0))
        print(f"    sat{slot['sat_id']}  {slot['t_start']:.1f}s–{slot['t_end']:.1f}s  "
              f"elev={slot['elev_deg']:.1f}°  delay={slot['delay_ms']:.1f} ms  "
              f"atm={atm_total:.2f}dB  "
              f"PER={slot['per']:.3f}{gap_str}")

    # Weighted-mean service-link delay (P2P channel delay is immutable)
    if schedule:
        total_dur = schedule[-1]["t_end"] - schedule[0]["t_start"]
        weighted_delay_ms = sum(
            s["delay_ms"] * (s["t_end"] - s["t_start"]) for s in schedule
        ) / max(total_dur, 1e-9)
    else:
        weighted_delay_ms = _one_way_delay_ms(SAT_HEIGHT_M, 60.0)

    first = schedule[0] if schedule else dict(
        delay_ms=weighted_delay_ms, per=0.01, elev_deg=60.0,
        sat_id=0, interruption_ms=0.0,
    )
    print(f"  Service link delay (weighted-mean): {weighted_delay_ms:.2f} ms")
    print(f"  Sat→Server:  rate={SAT_SERVER_DATARATE}  delay={BACKHAUL_DELAY_MS:.0f} ms")

    # =========================================================================
    # Node creation
    # =========================================================================
    # Node layout:
    #   [0 … num_clients-1] : phone nodes
    #   [num_clients]        : beam gateway (aggregation point for the beam)
    #   [num_clients+1]      : satellite (highest-elevation)
    #   [num_clients+2]      : internet server

    beam_gw_idx = num_clients
    sat_idx     = num_clients + 1
    server_idx  = num_clients + 2
    total_nodes = num_clients + 3

    nodes = ns.NodeContainer()
    nodes.Create(total_nodes)
    ns.InternetStackHelper().Install(nodes)

    phones  = [nodes.Get(i) for i in range(num_clients)]
    beam_gw = nodes.Get(beam_gw_idx)
    sat     = nodes.Get(sat_idx)
    server  = nodes.Get(server_idx)

    p2p = ns.PointToPointHelper()

    # ── Client positions ──────────────────────────────────────────────────────
    # Phones are placed on ground-level cells of the Munich scene (streets,
    # squares, courtyards) — never on top of buildings.  The walkable cells
    # are precomputed by build_walkable_points() during the RT stage and
    # loaded here from a pickle.  Stationary phones sit on a random walkable
    # cell; moving phones walk a RandomWaypoint path through a LOCAL subset
    # of walkable cells to reduce the chance of straight-line segments
    # cutting through buildings.
    _rng = _random_mod.Random(PY_RANDOM_SEED)

    _walkable_pts = []
    _walkable_path = "output/.walkable_points.pkl"
    if os.path.exists(_walkable_path):
        import pickle as _pickle
        with open(_walkable_path, "rb") as _f:
            _walkable_pts = _pickle.load(_f)
    if not _walkable_pts:
        print(f"[NS-3]  WARNING: no walkable points at {_walkable_path} — "
              f"falling back to open-area circle placement.")

    _walkable_arr = (np.asarray([(p[0], p[1]) for p in _walkable_pts],
                                 dtype=float)
                     if _walkable_pts else None)

    # Fast lookup: bucket walkable cells on the 5-m sampling grid so that
    # line-segment validation can be a single set membership check per sample.
    _WALK_GRID_SPACING = 5.0
    _walkable_grid = set()
    if _walkable_arr is not None:
        for _wx, _wy in _walkable_arr:
            _walkable_grid.add((int(round(_wx / _WALK_GRID_SPACING)),
                                 int(round(_wy / _WALK_GRID_SPACING))))

    def _is_walkable(x: float, y: float) -> bool:
        """Nearest-grid-cell test against the precomputed walkable set."""
        if not _walkable_grid:
            return True
        key = (int(round(x / _WALK_GRID_SPACING)),
               int(round(y / _WALK_GRID_SPACING)))
        return key in _walkable_grid

    def _segment_on_walkable(x0: float, y0: float,
                              x1: float, y1: float,
                              step: float = 2.0) -> bool:
        """Return True iff the straight segment from (x0,y0) to (x1,y1)
        passes only through walkable cells (sampled every `step` m)."""
        dx = x1 - x0
        dy = y1 - y0
        length = _math.hypot(dx, dy)
        n = max(1, int(length / step))
        for k in range(1, n + 1):
            a = k / n
            if not _is_walkable(x0 + a * dx, y0 + a * dy):
                return False
        return True

    def _random_pos_in_circle(radius, z=1.5):
        """Uniform random (x,y) within a circle of given radius, height z."""
        angle  = _rng.uniform(0, 2 * _math.pi)
        r      = radius * _math.sqrt(_rng.uniform(0, 1))
        return r * _math.cos(angle), r * _math.sin(angle), z

    def _random_walkable_pos(radius: float, z: float = 1.5):
        """
        Random ground-level cell within `radius` of the scene origin.
        Falls back to _random_pos_in_circle when no walkable grid is loaded.
        """
        if _walkable_arr is None:
            return _random_pos_in_circle(radius, z)
        r2 = _walkable_arr[:, 0] ** 2 + _walkable_arr[:, 1] ** 2
        mask = r2 <= radius * radius
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return _random_pos_in_circle(radius, z)
        i = idx[_rng.randrange(len(idx))]
        return float(_walkable_arr[i, 0]), float(_walkable_arr[i, 1]), z

    def _local_walkable_subset(cx: float, cy: float,
                                local_r: float, max_pts: int = 32) -> list:
        """
        Return a chained sequence of walkable (x,y) waypoints within
        `local_r` of (cx, cy).  The RandomWaypointMobilityModel cycles
        through a ListPositionAllocator, so each consecutive pair of
        waypoints forms a straight-line segment.  To avoid segments that
        clip buildings, waypoints are built as a greedy chain: each new
        waypoint is selected so that the line from the *previous* chosen
        waypoint stays on walkable cells.  The chain also closes back to
        the first waypoint (which is the allocator's cycle boundary).
        """
        if _walkable_arr is None:
            return []
        dx = _walkable_arr[:, 0] - cx
        dy = _walkable_arr[:, 1] - cy
        d2 = dx * dx + dy * dy

        r = local_r
        for _attempt in range(5):
            mask = d2 <= r * r
            idx = np.where(mask)[0]
            if len(idx) >= 8:
                break
            r *= 1.5
        if len(idx) == 0:
            return []

        cand = [(float(_walkable_arr[i, 0]), float(_walkable_arr[i, 1]))
                for i in idx]
        _rng.shuffle(cand)

        chain = []
        prev_x, prev_y = cx, cy
        for (wx, wy) in cand:
            if len(chain) >= max_pts:
                break
            if _segment_on_walkable(prev_x, prev_y, wx, wy):
                chain.append((wx, wy))
                prev_x, prev_y = wx, wy

        if len(chain) >= 2:
            # Ensure the cycle-close segment (last → first) is also clean
            # by rotating the chain until the close-back is walkable; if
            # no rotation works, drop the final waypoint.
            for _ in range(len(chain)):
                cx0, cy0 = chain[0]
                cxN, cyN = chain[-1]
                if _segment_on_walkable(cxN, cyN, cx0, cy0):
                    break
                chain.append(chain.pop(0))
            else:
                chain.pop()

        return chain

    # Each NS-3 Node may hold only ONE MobilityModel in its aggregation.
    # Object::DoGetObject returns the FIRST aggregated object whose TypeId
    # (or any parent TypeId) matches the requested type.  If we install
    # ConstantPositionMobilityModel on every node and then AggregateObject a
    # second MobilityModel (RWP on moving phones, ConstantVelocity on sat),
    # later GetObject<MobilityModel>() calls return the static one — so
    # positions never update.  Install exactly one mobility model per node.
    if NUM_PEDESTRIAN_MOVING_CLIENTS + NUM_VEHICULAR_MOVING_CLIENTS != NUM_MOVING_CLIENTS:
        raise ValueError(
            "NUM_PEDESTRIAN_MOVING_CLIENTS + NUM_VEHICULAR_MOVING_CLIENTS "
            "must equal NUM_MOVING_CLIENTS"
        )

    moving_start = NUM_STATIONARY_CLIENTS
    pedestrian_end = moving_start + NUM_PEDESTRIAN_MOVING_CLIENTS

    def _speed_bounds_for_client(client_idx: int) -> tuple:
        if client_idx < pedestrian_end:
            return PEDESTRIAN_SPEED_MIN_MS, PEDESTRIAN_SPEED_MAX_MS
        return VEHICULAR_SPEED_MIN_MS, VEHICULAR_SPEED_MAX_MS

    # Container of nodes that stay put (stationary phones + infrastructure).
    static_nc = ns.NodeContainer()
    for i in range(NUM_STATIONARY_CLIENTS):
        static_nc.Add(phones[i])
    static_nc.Add(beam_gw)
    static_nc.Add(server)

    static_mob = ns.MobilityHelper()
    static_mob.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    static_mob.Install(static_nc)

    # Place stationary phones at random walkable cells within the service
    # area.  Falls back to _random_pos_in_circle if no walkable grid is
    # loaded (e.g. first run before RT has populated the pickle).
    phone_positions = [None] * num_clients
    for i in range(NUM_STATIONARY_CLIENTS):
        px, py, pz = _random_walkable_pos(CLIENT_AREA_RADIUS_M)
        phone_positions[i] = (px, py, pz)
        phones[i].GetObject[ns.ConstantPositionMobilityModel]().SetPosition(
            ns.Vector(px, py, pz))

    # Moving phones: install RandomWaypointMobilityModel as the SOLE mobility
    # model on each node.  Waypoints are drawn from a LOCAL subset of
    # walkable cells around the phone's starting position (a pre-shuffled
    # ListPositionAllocator).  Keeping the subset small (~25 m pedestrians,
    # ~50 m vehicles) greatly reduces the chance that the straight-line
    # segment between two waypoints clips through a building, since
    # waypoints tend to stay within one street segment.
    for i in range(NUM_STATIONARY_CLIENTS, num_clients):
        px, py, pz = _random_walkable_pos(CLIENT_AREA_RADIUS_M)
        phone_positions[i] = (px, py, pz)
        rwp_mob = ns.CreateObject[ns.RandomWaypointMobilityModel]()
        vmin, vmax = _speed_bounds_for_client(i)
        is_pedestrian = (i < pedestrian_end)
        local_r = 25.0 if is_pedestrian else 50.0
        speed_var = ns.StringValue(
            f"ns3::UniformRandomVariable[Min={vmin}|Max={vmax}]"
        )
        pause_var = ns.StringValue("ns3::ConstantRandomVariable[Constant=0]")
        pos_alloc = ns.CreateObject[ns.ListPositionAllocator]()
        waypoints = _local_walkable_subset(px, py, local_r, max_pts=64)
        if not waypoints:
            # No walkable grid loaded — fall back to an unconstrained
            # rectangle in [-CLIENT_AREA_RADIUS_M, +CLIENT_AREA_RADIUS_M]².
            pos_alloc = ns.CreateObject[ns.RandomRectanglePositionAllocator]()
            r = CLIENT_AREA_RADIUS_M
            pos_alloc.SetAttribute("X", ns.StringValue(
                f"ns3::UniformRandomVariable[Min={-r}|Max={r}]"))
            pos_alloc.SetAttribute("Y", ns.StringValue(
                f"ns3::UniformRandomVariable[Min={-r}|Max={r}]"))
        else:
            for (wx, wy) in waypoints:
                pos_alloc.Add(ns.Vector(wx, wy, pz))
        rwp_mob.SetAttribute("Speed",             speed_var)
        rwp_mob.SetAttribute("Pause",             pause_var)
        rwp_mob.SetAttribute("PositionAllocator", ns.PointerValue(pos_alloc))
        phones[i].AggregateObject(rwp_mob)
        rwp_mob.SetPosition(ns.Vector(px, py, pz))

    # Satellite: ConstantVelocityMobilityModel as the SOLE mobility model.
    sat_mob = ns.CreateObject[ns.ConstantVelocityMobilityModel]()
    sat.AggregateObject(sat_mob)
    sat_mob.SetPosition(ns.Vector(0.0, 0.0, SAT_HEIGHT_M))
    sat_mob.SetVelocity(ns.Vector(SAT_ORBITAL_VELOCITY_MS, 0.0, 0.0))

    # ── IP address allocator ──────────────────────────────────────────────────
    ipv4 = ns.Ipv4AddressHelper()
    subnet_iter = [0]  # mutable counter for subnet numbering

    def _next_subnet():
        n = subnet_iter[0]
        subnet_iter[0] += 1
        return f"10.{n // 256}.{n % 256}.0"

    # =========================================================================
    # Wiring up links
    # =========================================================================

    # ── Phone → beam gateway (per-UE access, no contention) ──────────────────
    # Each phone has its own p2p to the aggregation gateway at a high rate so
    # that the access link is never the bottleneck.  This stands in for the
    # OFDMA/scheduled allocation within one spot beam: individual UEs don't
    # contend for the air interface (the satellite schedules them).
    access_p2p = ns.PointToPointHelper()
    access_p2p.SetDeviceAttribute("DataRate", ns.StringValue("1Gbps"))
    access_p2p.SetChannelAttribute("Delay",    ns.StringValue("0.01ms"))
    # Map each phone's source IP (as 32-bit integer via Ipv4Address::Get)
    # to its traffic profile so FlowMonitor records can be bucketed into
    # profile_stats by the actual sender, not by round-robin over profile
    # names (which scrambled profiles with 50× different bitrates).
    ip_to_profile = {}
    for ci, ph in enumerate(phones):
        devs_acc = access_p2p.Install(_nc2(ph, beam_gw))
        ipv4.SetBase(ns.Ipv4Address(_next_subnet()),
                     ns.Ipv4Mask("255.255.255.0"))
        ifaces_acc = ipv4.Assign(devs_acc)
        # Device index 0 is the phone side of the p2p.
        pname = CLIENT_PROFILES[ci] if ci < len(CLIENT_PROFILES) else "bulk"
        ip_to_profile[int(ifaces_acc.GetAddress(0).Get())] = pname

    # ── Beam gateway → satellite shared service link (true bottleneck) ───────
    # A single p2p at SERVICE_LINK_RATE_MBPS models the aggregate capacity of
    # one LEO spot beam: all UE flows funnel through this queue, so the
    # aggregate cannot exceed the beam rate and TCP congestion/queuing dynamics
    # are exercised at the correct bandwidth.  A RateErrorModel on the
    # satellite's RX device applies the RT-calibrated PER uniformly — all UEs
    # within a ~500 m cell see the same fading to a 550 km LEO satellite.
    svc_p2p = ns.PointToPointHelper()
    svc_p2p.SetDeviceAttribute("DataRate",
        ns.StringValue(f"{SERVICE_LINK_RATE_MBPS}Mbps"))
    svc_p2p.SetChannelAttribute("Delay",
        ns.StringValue(f"{weighted_delay_ms:.3f}ms"))
    devs_svc = svc_p2p.Install(_nc2(beam_gw, sat))

    em_svc = ns.CreateObject[ns.RateErrorModel]()
    em_svc.SetAttribute("ErrorRate", ns.DoubleValue(first["per"]))
    em_svc.SetAttribute("ErrorUnit", ns.StringValue("ERROR_UNIT_PACKET"))
    # Satellite is device index 1 on this p2p — apply fading to its RX.
    devs_svc.Get(1).SetAttribute(
        "ReceiveErrorModel", ns.PointerValue(em_svc))

    # ── N1 fix: burst-correlated fading state machine ─────────────────────────
    # Shadow fading has a coherence time of ~FADE_MEAN_DURATION_MS.
    # The plain RateErrorModel drops packets i.i.d., which lets TCP recover
    # quickly via fast-retransmit.  Real bursty fading triggers RTO.
    # This state machine models that by clustering losses into exponentially-
    # distributed burst events rather than independent per-packet coin flips.
    # The fade entry rate is chosen so that the long-run average PER matches
    # the link-budget PER: rate_in × fade_mean = PER  →  rate_in = PER / T_fade.
    _FADE_MEAN_S = FADE_MEAN_DURATION_MS / 1000.0
    _fade_rng = _random_mod.Random(PY_RANDOM_SEED + 17)
    _fade_active = [False]
    _slot_per    = [first["per"]]   # updated at each handover restore

    def _schedule_next_fade():
        per_now = _slot_per[0]
        if per_now < 1e-4:
            return
        rate = per_now / _FADE_MEAN_S   # fades per second
        wait = _fade_rng.expovariate(rate)
        ev   = ns.cppyy.gbl.pythonMakeEvent(_enter_fade)
        ns.Simulator.Schedule(ns.Seconds(wait), ev)

    def _enter_fade():
        if _fade_active[0]:
            return
        _fade_active[0] = True
        em_svc.SetAttribute("ErrorRate", ns.DoubleValue(1.0))
        dur = max(0.005, _fade_rng.expovariate(1.0 / _FADE_MEAN_S))
        ev  = ns.cppyy.gbl.pythonMakeEvent(_exit_fade)
        ns.Simulator.Schedule(ns.Seconds(dur), ev)

    def _exit_fade():
        _fade_active[0] = False
        em_svc.SetAttribute("ErrorRate", ns.DoubleValue(_slot_per[0]))
        _schedule_next_fade()

    _schedule_next_fade()

    ipv4.SetBase(ns.Ipv4Address(_next_subnet()),
                 ns.Ipv4Mask("255.255.255.0"))
    ipv4.Assign(devs_svc)

    # ── Satellite → Internet Server (backhaul: feeder + gateway + transit) ────
    p2p.SetDeviceAttribute("DataRate",  ns.StringValue(SAT_SERVER_DATARATE))
    p2p.SetChannelAttribute("Delay",    ns.StringValue(f"{BACKHAUL_DELAY_MS}ms"))
    devs_srv = p2p.Install(_nc2(sat, server))
    ipv4.SetBase(ns.Ipv4Address(_next_subnet()), ns.Ipv4Mask("255.255.255.0"))
    iface_srv = ipv4.Assign(devs_srv)

    ns.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

    # =========================================================================
    # Beam management handover scheduling
    # =========================================================================
    # Each handover has two scheduled events:
    #   Event A  at slot["t_start"]                    → blackout (PER = 1.0)
    #   Event B  at slot["t_start"] + gap_ms / 1000   → restore  (PER = slot["per"])
    #
    # All service-link error models in em_svc_list are updated together
    # because all clients are within CLIENT_AREA_RADIUS_M of each other,
    # which is negligible vs the satellite footprint (~100 km).

    handover_queue = collections.deque(schedule[1:])
    handover_count = [0]
    handover_times = []   # list of (t_blackout_start, t_blackout_end) tuples

    def _handover_blackout():
        """Phase 1: set PER = 1.0 (beam failure / link blackout)."""
        if not handover_queue:
            return
        slot   = handover_queue[0]   # peek — don't pop yet
        gap_s  = slot["interruption_ms"] / 1000.0
        t_now  = ns.Simulator.Now().GetSeconds()
        handover_times.append((t_now, t_now + gap_s))
        print(f"  [t={t_now:.2f}s] Handover blackout → sat{slot['sat_id']}  "
              f"gap={slot['interruption_ms']:.0f} ms  "
              f"elev={slot['elev_deg']:.1f}°")
        # Single shared beam → one error model covers all UEs
        em_svc.SetAttribute("ErrorRate", ns.DoubleValue(1.0))
        # Schedule the restore event after the gap
        ev = ns.cppyy.gbl.pythonMakeEvent(_handover_restore)
        ns.Simulator.Schedule(ns.Seconds(gap_s), ev)

    def _handover_restore():
        """Phase 2: restore PER = slot['per'] after beam management gap."""
        if not handover_queue:
            return
        slot = handover_queue.popleft()
        _slot_per[0] = slot["per"]   # keep burst-fade state machine in sync
        _fade_active[0] = False      # end any in-progress fade on the old sat
        em_svc.SetAttribute("ErrorRate", ns.DoubleValue(slot["per"]))
        handover_count[0] += 1
        print(f"  [t={ns.Simulator.Now().GetSeconds():.2f}s] "
              f"Handover restored → sat{slot['sat_id']}  "
              f"PER={slot['per']:.3f}")
        _schedule_next_fade()        # restart fade scheduling for new satellite
        # Schedule the next handover blackout if one exists
        if handover_queue:
            nxt = handover_queue[0]
            if nxt["t_start"] < SIM_DURATION_S:
                ev = ns.cppyy.gbl.pythonMakeEvent(_handover_blackout)
                ns.Simulator.Schedule(
                    ns.Seconds(nxt["t_start"] - ns.Simulator.Now().GetSeconds()),
                    ev)

    if len(schedule) > 1:
        first_ho = schedule[1]
        if first_ho["t_start"] < SIM_DURATION_S:
            ev = ns.cppyy.gbl.pythonMakeEvent(_handover_blackout)
            ns.Simulator.Schedule(ns.Seconds(first_ho["t_start"]), ev)

    # =========================================================================
    # Application layer — transport comparison modes
    # =========================================================================
    base_port = 9000
    profile_names = list(TRAFFIC_PROFILES.keys())
    profile_ports = {p: base_port + i for i, p in enumerate(profile_names)}

    # Compute per-flow MaxBytes for TCP BulkSend
    max_bytes = 0
    if DATA_VOLUME_MB > 0:
        max_bytes = int(DATA_VOLUME_MB * 1_000_000)

    udp_sinks_by_profile = {}
    tcp_sinks_by_profile = {}
    for pname, pport in profile_ports.items():
        udp_sink = ns.PacketSinkHelper(
            "ns3::UdpSocketFactory",
            ns.InetSocketAddress(ns.Ipv4Address.GetAny(), pport).ConvertTo())
        tcp_sink = ns.PacketSinkHelper(
            "ns3::TcpSocketFactory",
            ns.InetSocketAddress(ns.Ipv4Address.GetAny(), pport).ConvertTo())
        udp_app = udp_sink.Install(server)
        udp_app.Start(ns.Seconds(0.0))
        udp_app.Stop(ns.Seconds(SIM_DURATION_S + 2.0))
        tcp_app = tcp_sink.Install(server)
        tcp_app.Start(ns.Seconds(0.0))
        tcp_app.Stop(ns.Seconds(SIM_DURATION_S + 2.0))
        udp_sinks_by_profile[pname] = udp_app.Get(0).GetObject[ns.PacketSink]()
        tcp_sinks_by_profile[pname] = tcp_app.Get(0).GetObject[ns.PacketSink]()

    if TRANSPORT_COMPARE_MODE == "matched_offered_load":
        active_ip_protos = {6} if ns3_proto == "tcp" else {17}
    else:
        # Legacy mode keeps UDP profiles even in TCP runs.
        active_ip_protos = {17}
        if ns3_proto == "tcp":
            active_ip_protos.add(6)

    def _profile_rate(profile_name: str, profile: dict) -> str:
        if TRANSPORT_COMPARE_MODE == "matched_offered_load":
            return MATCHED_PROFILE_DATA_RATES.get(profile_name, APP_DATA_RATE)
        return profile.get("data_rate") or APP_DATA_RATE

    for ci, phone in enumerate(phones):
        profile_name = CLIENT_PROFILES[ci] if ci < len(CLIENT_PROFILES) else "bulk"
        profile      = TRAFFIC_PROFILES[profile_name]
        app_type     = profile["app_type"]
        pkt_size     = int(profile.get("packet_size", PACKET_SIZE_BYTES))
        duty         = float(profile.get("duty", 1.0))
        server_addr  = ns.InetSocketAddress(
            iface_srv.GetAddress(1), profile_ports[profile_name]
        )

        if duty >= 1.0:
            on_time_str  = "ns3::ConstantRandomVariable[Constant=1]"
            off_time_str = "ns3::ConstantRandomVariable[Constant=0]"
        else:
            on_time_str  = "ns3::UniformRandomVariable[Min=0.9|Max=1.1]"
            off_period   = (1.0 / max(duty, 0.01)) - 1.0
            off_time_str = (
                "ns3::UniformRandomVariable"
                f"[Min={off_period*0.9:.2f}|Max={off_period*1.1:.2f}]"
            )

        if TRANSPORT_COMPARE_MODE == "matched_offered_load":
            socket_factory = (
                "ns3::TcpSocketFactory" if ns3_proto == "tcp"
                else "ns3::UdpSocketFactory"
            )
            onoff = ns.OnOffHelper(socket_factory, server_addr.ConvertTo())
            onoff.SetAttribute("DataRate",   ns.StringValue(_profile_rate(profile_name, profile)))
            onoff.SetAttribute("PacketSize", ns.UintegerValue(pkt_size))
            onoff.SetAttribute("OnTime",     ns.StringValue(on_time_str))
            onoff.SetAttribute("OffTime",    ns.StringValue(off_time_str))
            for _ in range(NUM_PARALLEL_FLOWS):
                app_tx = onoff.Install(phone)
                app_tx.Start(ns.Seconds(1.0))
                app_tx.Stop(ns.Seconds(SIM_DURATION_S))
        else:
            # Legacy behavior: bulk uses TCP BulkSend in TCP runs, otherwise CBR.
            if ns3_proto != "tcp" or app_type != "tcp_bulk":
                onoff = ns.OnOffHelper(
                    "ns3::UdpSocketFactory", server_addr.ConvertTo())
                onoff.SetAttribute("DataRate",   ns.StringValue(_profile_rate(profile_name, profile)))
                onoff.SetAttribute("PacketSize", ns.UintegerValue(pkt_size))
                onoff.SetAttribute("OnTime",     ns.StringValue(on_time_str))
                onoff.SetAttribute("OffTime",    ns.StringValue(off_time_str))
                for _ in range(NUM_PARALLEL_FLOWS):
                    app_tx = onoff.Install(phone)
                    app_tx.Start(ns.Seconds(1.0))
                    app_tx.Stop(ns.Seconds(SIM_DURATION_S))
            else:
                bulk = ns.BulkSendHelper(
                    "ns3::TcpSocketFactory", server_addr.ConvertTo())
                bulk.SetAttribute("SendSize", ns.UintegerValue(pkt_size))
                bulk.SetAttribute("MaxBytes", ns.UintegerValue(max_bytes))
                for _ in range(NUM_PARALLEL_FLOWS):
                    app_tx = bulk.Install(phone)
                    app_tx.Start(ns.Seconds(1.0))
                    app_tx.Stop(ns.Seconds(SIM_DURATION_S))

    # =========================================================================
    # Per-second throughput time-series probe (aggregate sinks)
    # =========================================================================
    ts_t_list     = []
    ts_bytes_list = []
    def _all_sinks_total_rx() -> int:
        total = 0
        for pname in profile_names:
            total += int(udp_sinks_by_profile[pname].GetTotalRx())
            total += int(tcp_sinks_by_profile[pname].GetTotalRx())
        return total

    def _streaming_sink_total_rx() -> int:
        if "streaming" not in udp_sinks_by_profile:
            return 0
        return (
            int(udp_sinks_by_profile["streaming"].GetTotalRx())
            + int(tcp_sinks_by_profile["streaming"].GetTotalRx())
        )

    ts_stream_bytes_list = []

    def _ts_probe():
        t = ns.Simulator.Now().GetSeconds()
        cum_bytes = _all_sinks_total_rx()
        stream_bytes = _streaming_sink_total_rx()
        ts_t_list.append(t)
        ts_bytes_list.append(int(cum_bytes))
        ts_stream_bytes_list.append(int(stream_bytes))
        if t + TIMESERIES_BUCKET_S <= SIM_DURATION_S + 1.0:
            ev = ns.cppyy.gbl.pythonMakeEvent(_ts_probe)
            ns.Simulator.Schedule(ns.Seconds(TIMESERIES_BUCKET_S), ev)

    # Start first probe at t = TIMESERIES_BUCKET_S
    ev_ts = ns.cppyy.gbl.pythonMakeEvent(_ts_probe)
    ns.Simulator.Schedule(ns.Seconds(TIMESERIES_BUCKET_S), ev_ts)

    # =========================================================================
    # Mobility position-trace probe (Phase A — optional)
    # =========================================================================
    # When sample_positions is True, sample every phone's (x,y,z) and the
    # satellite's (x,y,z) every MOBILITY_SAMPLE_DT_S seconds.  Stored in
    # result["position_trace"] for the mobility-video renderer.  Only the
    # first NS-3 worker runs this probe (all workers share the same
    # Random(42) mobility seed, so one trace is representative).
    MOBILITY_SAMPLE_DT_S = 0.1
    pos_t_list     = []
    pos_phone_list = []   # list of [(x,y,z), ...] per timestep
    pos_sat_list   = []   # list of (x,y,z) per timestep

    if sample_positions:
        def _pos_probe():
            t = ns.Simulator.Now().GetSeconds()
            pos_t_list.append(round(t, 3))
            frame = []
            for ph in phones:
                p = ph.GetObject[ns.MobilityModel]().GetPosition()
                frame.append((round(p.x, 2), round(p.y, 2), round(p.z, 2)))
            pos_phone_list.append(frame)
            sp = sat.GetObject[ns.MobilityModel]().GetPosition()
            pos_sat_list.append((round(sp.x, 2), round(sp.y, 2), round(sp.z, 2)))
            if t + MOBILITY_SAMPLE_DT_S <= SIM_DURATION_S + 1e-6:
                ev = ns.cppyy.gbl.pythonMakeEvent(_pos_probe)
                ns.Simulator.Schedule(ns.Seconds(MOBILITY_SAMPLE_DT_S), ev)

        ev_pos = ns.cppyy.gbl.pythonMakeEvent(_pos_probe)
        ns.Simulator.Schedule(ns.Seconds(0.0), ev_pos)

    # ── FlowMonitor ───────────────────────────────────────────────────────────
    fm_helper = ns.FlowMonitorHelper()
    endpoint_nc = ns.NodeContainer()
    for ph in phones:
        endpoint_nc.Add(ph)
    endpoint_nc.Add(server)
    monitor = fm_helper.Install(endpoint_nc)

    # ── Run simulation ────────────────────────────────────────────────────────
    ns.Simulator.Stop(ns.Seconds(SIM_DURATION_S + 3.0))
    ns.Simulator.Run()
    monitor.CheckForLostPackets()

    # ── Build per-second time-series ──────────────────────────────────────────
    ts_throughput_kbps = []
    ts_streaming_kbps = []
    for k in range(len(ts_t_list)):
        prev_bytes = ts_bytes_list[k - 1] if k > 0 else 0
        delta_bytes = max(ts_bytes_list[k] - prev_bytes, 0)
        prev_stream_bytes = ts_stream_bytes_list[k - 1] if k > 0 else 0
        delta_stream_bytes = max(ts_stream_bytes_list[k] - prev_stream_bytes, 0)
        ts_throughput_kbps.append(round(delta_bytes * 8.0 / TIMESERIES_BUCKET_S / 1e3, 2))
        ts_streaming_kbps.append(round(delta_stream_bytes * 8.0 / TIMESERIES_BUCKET_S / 1e3, 2))

    timeseries = {
        "t_s":              [round(t, 1) for t in ts_t_list],
        "throughput_kbps":  ts_throughput_kbps,
        "streaming_throughput_kbps": ts_streaming_kbps,
        "handover_times":   list(handover_times),
    }

    # Position trace (Phase A — only when sample_positions=True)
    position_trace = None
    if sample_positions and pos_t_list:
        position_trace = {
            "t_s":       pos_t_list,
            "phones":    pos_phone_list,
            "satellite": pos_sat_list,
            "dt_s":      MOBILITY_SAMPLE_DT_S,
            "num_stationary": NUM_STATIONARY_CLIENTS,
            "num_pedestrian": NUM_PEDESTRIAN_MOVING_CLIENTS,
            "num_vehicular":  NUM_VEHICULAR_MOVING_CLIENTS,
        }
        print(f"  [Mobility]  Sampled {len(pos_t_list)} frames "
              f"({1.0/MOBILITY_SAMPLE_DT_S:.0f} Hz, {SIM_DURATION_S:.1f} s)")

    # ── Collect FlowMonitor statistics ────────────────────────────────────────
    stats      = monitor.GetFlowStats()
    classifier = fm_helper.GetClassifier()

    result = dict(
        scenario        = scenario,
        protocol        = proto,
        label           = label,
        topology        = "direct",
        elevation_deg   = first["elev_deg"],
        svc_delay_ms    = round(weighted_delay_ms, 2),
        svc_loss_pct    = round(first["per"] * 100.0, 2),
        tx_packets      = 0,
        rx_packets      = 0,
        loss_pct        = 0.0,
        mean_delay_ms   = 0.0,
        jitter_ms       = 0.0,
        throughput_kbps = 0.0,
        handovers       = handover_count[0],
        schedule        = schedule,
        fairness_index  = 0.0,
        profile_stats   = {p: {"tx": 0, "rx": 0, "rx_bytes": 0, "delay_sum": 0.0}
                           for p in TRAFFIC_PROFILES},
        timeseries      = timeseries,
        position_trace  = position_trace,
        is_analytical   = False,
    )

    active_s = SIM_DURATION_S - 1.0   # exclude 1 s warm-up

    total_rx   = 0
    total_tx   = 0
    delay_sum  = 0.0
    jitter_sum = 0.0
    rx_bytes   = 0
    flow_tputs = []   # per-flow throughput for Jain's fairness

    # Profile tagging: map FlowMonitor records to profiles via destination port
    # (preferred; one sink port per profile), then source IP fallback.
    port_to_profile = {int(p): n for n, p in profile_ports.items()}
    tcp_flow_count = 0

    for pair in stats:
        fid, fs = pair.first, pair.second
        finfo   = classifier.FindFlow(fid)
        ip_proto = finfo.protocol

        if ip_proto not in active_ip_protos:
            continue
        if fs.rxPackets == 0:
            continue

        rx_n   = int(fs.rxPackets)
        tx_n   = int(fs.txPackets)
        d_s    = fs.delaySum.GetSeconds()
        j_s    = fs.jitterSum.GetSeconds()
        b      = int(fs.rxBytes)
        flow_tput = b * 8.0 / active_s / 1e3  # kbps; fixed denominator → correct Jain's index

        total_rx   += rx_n
        total_tx   += tx_n
        delay_sum  += d_s
        jitter_sum += j_s
        rx_bytes   += b
        flow_tputs.append(flow_tput)

        if ip_proto == 6:
            tcp_flow_count += 1
        dst_port = int(getattr(finfo, "destinationPort", 0))
        pname = port_to_profile.get(dst_port)
        if pname is None:
            src_key = int(finfo.sourceAddress.Get())
            pname = ip_to_profile.get(src_key, "bulk")
        if pname not in result["profile_stats"]:
            pname = "bulk"
        ps = result["profile_stats"][pname]
        ps["tx"] += tx_n
        ps["rx"] += rx_n
        ps["rx_bytes"] += b
        ps["delay_sum"] += d_s

    if total_rx > 0:
        # Jain's fairness index: J = (Σx)² / (n · Σx²)
        n_flows = len(flow_tputs)
        sum_sq  = sum(x**2 for x in flow_tputs)
        fairness = (sum(flow_tputs)**2 / (n_flows * sum_sq)) if (n_flows > 0 and sum_sq > 0) else 0.0

        result.update(
            tx_packets      = total_tx,
            rx_packets      = total_rx,
            loss_pct        = round(100.0 * (1 - total_rx / max(total_tx, 1)), 2),
            mean_delay_ms   = round(delay_sum / total_rx * 1e3, 2),
            jitter_ms       = round(jitter_sum / max(total_rx - 1, 1) * 1e3, 2),
            throughput_kbps = round(rx_bytes * 8.0 / active_s / 1e3, 2),
            fairness_index  = round(fairness, 4),
        )

    # ── Protocol overhead (L3/L4 header bytes vs payload bytes) ───────────────
    # FlowMonitor's rxBytes is the IP-payload byte count (UDP or TCP).
    # The L3/L4 wire overhead per packet is:
    #   UDP:  20 (IPv4) + 8  (UDP) = 28 bytes / packet
    #   TCP:  20 (IPv4) + 20 (TCP) = 40 bytes / packet (no options)
    UDP_HEADER_BYTES = 28
    TCP_HEADER_BYTES = 40
    overhead_hdr_bytes  = UDP_HEADER_BYTES if ns3_proto != "tcp" else TCP_HEADER_BYTES
    total_hdr_bytes     = total_rx * overhead_hdr_bytes
    total_wire_bytes    = rx_bytes + total_hdr_bytes
    protocol_overhead_pct = (100.0 * total_hdr_bytes / total_wire_bytes
                              if total_wire_bytes > 0 else 0.0)
    result["protocol_overhead_pct"] = round(protocol_overhead_pct, 2)

    # ── Cwnd dynamics proxy (RFC 5681 BDP-derived) ────────────────────────────
    # Direct NS-3 cwnd tracing requires per-socket callbacks that are brittle
    # across cppyy versions.  Instead we derive an effective in-flight window
    # from the per-second throughput time-series using Little's law:
    #   cwnd_eff_bytes = throughput_bps × RTT_s
    # This is exact for a link that is neither queue- nor loss-limited (BDP).
    # Under loss or RTO events the time-series already captures the dip, so
    # the proxy cwnd trace faithfully reflects TCP recovery dynamics.
    mean_rtt_s_ts = max((result.get("mean_delay_ms", weighted_delay_ms) * 2.0) / 1e3,
                        1e-3)
    cwnd_eff_bytes = [
        int(kbps * 1e3 * mean_rtt_s_ts / 8.0)
        for kbps in ts_throughput_kbps
    ]
    result["timeseries"]["cwnd_eff_bytes"] = cwnd_eff_bytes
    result["timeseries"]["rtt_s"]          = round(mean_rtt_s_ts, 4)

    # ── Application-layer derived KPIs ────────────────────────────────────────
    # Rebuffering ratio: fraction of simulation seconds where aggregate
    # streaming throughput fell below 80% of the required target (sum of
    # all streaming clients' CBR bitrate).  Computed from the per-second
    # time-series already collected at TIMESERIES_BUCKET_S resolution.
    from config import TRAFFIC_PROFILE_COUNTS as _TPC
    streaming_clients = _TPC.get("streaming", 0)
    streaming_profile = TRAFFIC_PROFILES.get("streaming", {})
    streaming_bitrate_str = streaming_profile.get("data_rate", "0kbps")

    def _parse_bitrate_kbps(spec):
        if not spec:
            return 0.0
        s = str(spec).strip().lower()
        try:
            if s.endswith("gbps"): return float(s[:-4]) * 1e6
            if s.endswith("mbps"): return float(s[:-4]) * 1e3
            if s.endswith("kbps"): return float(s[:-4])
            if s.endswith("bps"):  return float(s[:-3]) / 1e3
        except ValueError:
            pass
        return 0.0

    streaming_per_client_kbps = _parse_bitrate_kbps(streaming_bitrate_str)
    streaming_target_kbps = streaming_per_client_kbps * streaming_clients

    rebuffer_threshold_kbps = 0.8 * streaming_target_kbps
    rebuffer_seconds = 0
    observed_seconds = 0
    ts_stream_for_qoe = timeseries.get("streaming_throughput_kbps", ts_throughput_kbps)
    for tt, kbps in zip(ts_t_list, ts_stream_for_qoe):
        if tt < 2.0:   # skip warm-up probe (app starts at t=1s)
            continue
        observed_seconds += 1
        if streaming_target_kbps > 0.0 and kbps < rebuffer_threshold_kbps:
            rebuffer_seconds += 1
    rebuffer_ratio = (rebuffer_seconds / observed_seconds
                      if observed_seconds > 0 else 0.0)

    # Video PSNR (streaming profile RX throughput → empirical PSNR map)
    stream_rx_bytes = result["profile_stats"]["streaming"]["rx_bytes"]
    stream_bitrate_kbps = stream_rx_bytes * 8.0 / active_s / 1e3
    per_client_stream_kbps = (stream_bitrate_kbps / streaming_clients
                               if streaming_clients > 0 else 0.0)
    stream_psnr_db = _video_psnr_db(per_client_stream_kbps)

    # Gaming threshold pass/fail (E2E latency < 50 ms AND loss < 1%)
    gaming_stats  = result["profile_stats"].get("gaming", {})
    gaming_rx     = gaming_stats.get("rx", 0)
    gaming_tx     = gaming_stats.get("tx", 0)
    gaming_delay  = gaming_stats.get("delay_sum", 0.0)
    gaming_latency_ms = (gaming_delay / gaming_rx * 1e3
                         if gaming_rx > 0 else float("inf"))
    gaming_loss_pct   = (100.0 * (1 - gaming_rx / max(gaming_tx, 1))
                         if gaming_tx > 0 else 100.0)
    gaming_pass = (gaming_latency_ms < 50.0 and gaming_loss_pct < 1.0)

    # VoIP MOS (ITU-T G.107 E-model)
    voice_stats   = result["profile_stats"].get("voice", {})
    voice_rx      = voice_stats.get("rx", 0)
    voice_tx      = voice_stats.get("tx", 0)
    voice_delay   = voice_stats.get("delay_sum", 0.0)
    voice_latency_ms = (voice_delay / voice_rx * 1e3
                        if voice_rx > 0 else 0.0)
    voice_loss_pct   = (100.0 * (1 - voice_rx / max(voice_tx, 1))
                        if voice_tx > 0 else 0.0)
    voice_mos = _g107_mos(voice_latency_ms, voice_loss_pct)

    result.update(
        rebuffer_ratio_pct   = round(100.0 * rebuffer_ratio, 2),
        rebuffer_seconds     = rebuffer_seconds,
        stream_psnr_db       = round(stream_psnr_db, 2),
        gaming_latency_ms    = round(gaming_latency_ms, 2)
                                if math.isfinite(gaming_latency_ms) else None,
        gaming_loss_pct      = round(gaming_loss_pct, 2),
        gaming_pass          = bool(gaming_pass),
        voice_mos            = round(voice_mos, 3),
        voice_latency_ms     = round(voice_latency_ms, 2),
        voice_loss_pct       = round(voice_loss_pct, 2),
    )

    # ── Handover success rate (3GPP TS 38.300 §10.1.2.3) ──────────────────────
    # A handover is deemed successful when data transfer resumes after the
    # scheduled blackout window — i.e., at least one per-second probe shows
    # non-zero throughput within HANDOVER_RECOVERY_WINDOW_S of the gap end.
    # A probe returning 0 kbps in every subsequent second indicates the UE
    # failed to reattach to the target beam within the recovery budget.
    ho_success = 0
    ho_total   = 0
    HANDOVER_RECOVERY_WINDOW_S = HANDOVER_INTERRUPTION_MS_MAX / 1000.0 + 1.0
    ts_t   = timeseries["t_s"]
    ts_bps = timeseries["throughput_kbps"]
    for (t_blk_start, t_blk_end) in handover_times:
        ho_total += 1
        recovered = False
        for tt, kbps in zip(ts_t, ts_bps):
            if t_blk_end <= tt <= t_blk_end + HANDOVER_RECOVERY_WINDOW_S:
                if kbps > 0.0:
                    recovered = True
                    break
        if recovered:
            ho_success += 1

    result["handover_events"]       = ho_total
    result["handover_successes"]    = ho_success
    result["handover_success_rate"] = (
        round(ho_success / ho_total, 4) if ho_total > 0 else 1.0
    )

    ns.Simulator.Destroy()

    print(f"\n  Results ({label}, direct):")
    for k, v in result.items():
        if k not in ("schedule", "profile_stats", "timeseries"):
            print(f"    {k:<24s}: {v}")
    print(f"    {'fairness_index':<24s}: {result['fairness_index']:.4f}")

    return result


# =============================================================================
# Parallel protocol runner helpers
# =============================================================================

def _ns3_parallel_worker(args: tuple) -> dict:
    """
    Top-level worker function for multiprocessing.Pool.

    Must be at module level (not nested) so it can be pickled by the
    spawn-context Pool.  Each worker runs in its own process with its
    own NS-3 instance, so there is no shared global state to worry about.
    """
    (scenario, pcfg, channel_stats, snr_thresh_db, sigmoid_slope,
     sample_positions, channel_stats_snapshots, snapshot_times) = args
    return run_ns3(scenario, pcfg, channel_stats, snr_thresh_db, sigmoid_slope,
                   sample_positions=sample_positions,
                   channel_stats_snapshots=channel_stats_snapshots,
                   snapshot_times=snapshot_times)


# =============================================================================
# Protocol suite runner — returns (results, results) for backward compat
# =============================================================================

def run_ns3_both_topologies(channel_stats: list,
                             scenario: str = "urban",
                             snr_thresh_db: float = None,
                             sigmoid_slope: float = None,
                             channel_stats_snapshots: list = None,
                             snapshot_times: list = None) -> tuple:
    """
    Run the NS-3 simulation once per entry in config.PROTOCOLS using the
    topology mode configured in config.USE_BASE_STATIONS, then return the
    results as a (direct_results, indirect_results) tuple for backward
    compatibility with main.py and topology_diagram.py.

    Because the topology is now determined by USE_BASE_STATIONS (a single
    configurable flag), the same result set is returned in both slots of
    the tuple.  All plot functions in topology_diagram.py continue to
    work without modification.

    For QUIC: the NS-3 simulation runs as UDP.  After the BBR run completes,
    RFC 9000 corrections are applied analytically via _apply_quic_corrections.

    Parameters
    ----------
    channel_stats : list[dict]
        RT channel statistics from rt_sim.run_ray_tracing().
    scenario : str
        NTN propagation scenario label (used for display only).
    snr_thresh_db : float, optional
        Sigmoid threshold [dB] fitted to the Sionna LDPC BER curve.
    sigmoid_slope : float, optional
        Sigmoid slope [1/dB] fitted to the Sionna LDPC BER curve.

    Returns
    -------
    tuple (results, results)
        Both elements contain the same list[dict], one entry per protocol.
        The tuple structure is kept for backward compatibility with callers
        that unpack as:  direct_results, indirect_results = run_ns3_both_topologies(...)
    """
    import multiprocessing as _mp

    common_tcp = dict(
        tcp_snd_buf    = TCP_SNDRCV_BUF_BYTES,
        tcp_rcv_buf    = TCP_SNDRCV_BUF_BYTES,
        tcp_sack       = TCP_SACK_ENABLED,
        tcp_timestamps = TCP_TIMESTAMPS,
    )

    # Build one args-tuple per non-QUIC protocol (QUIC is derived analytically).
    # Only the first worker samples mobility positions — the rest share the
    # same Random(42) seed so one trace is representative, and avoiding the
    # per-frame list appends in the other workers saves memory.
    non_quic_cfgs = [pcfg for pcfg in PROTOCOLS if pcfg["protocol"] != "quic"]
    args_list = [
        (scenario, {**common_tcp, **pcfg}, channel_stats,
         snr_thresh_db, sigmoid_slope, (i == 0),
         channel_stats_snapshots, snapshot_times)
        for i, pcfg in enumerate(non_quic_cfgs)
    ]

    # Each NS-3 run is single-threaded and independent; spawn one process per
    # protocol so all protocols execute simultaneously.  spawn (not fork) is
    # used for safety — it starts a clean Python interpreter that re-imports
    # ntn_ns3 (and therefore NS-3) fresh, avoiding shared-library state issues.
    #
    # Stdout from parallel workers is interleaved; redirect to a file if you
    # need clean per-protocol logs:  python main.py 2>&1 | tee run.log
    n_workers = min(len(args_list), os.cpu_count() or 4)
    print(f"\n  Spawning {n_workers} parallel NS-3 workers "
          f"({len(args_list)} protocols) ...")
    ctx = _mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = list(pool.map(_ns3_parallel_worker, args_list))

    # QUIC: analytical corrections on top of BBR baseline (main process only)
    bbr_result = next((r for r in results if r.get("label") == "TCP BBR"), None)
    if bbr_result is not None:
        quic_result = _apply_quic_corrections(
            bbr_result,
            bbr_result["schedule"],
            link_rate_bps=SERVICE_LINK_RATE_MBPS * 1e6,
        )
        quic_result["topology"] = "direct"
        results.append(quic_result)
    else:
        print("[QUIC]  No BBR result found — skipping QUIC.")

    # Return the same result set in both slots for backward compatibility
    return results, results


# =============================================================================
# Legacy single-topology runner (backward compatibility)
# =============================================================================

def run_ns3_protocol_suite(channel_stats: list,
                            scenario: str = "urban") -> list:
    """Kept for backward compatibility. Use run_ns3_both_topologies instead."""
    results, _ = run_ns3_both_topologies(channel_stats, scenario)
    return results
