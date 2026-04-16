"""
plots/__init__.py — Re-exports all draw_* functions and PROTO_COLORS.

Importing `from plots import draw_X` continues to work unchanged.
"""

from plots.phy import draw_ber_bler
from plots.channel import draw_link_budget_waterfall, draw_snr_vs_elevation, draw_channel_validation
from plots.network import (
    draw_protocol_comparison,
    draw_latency_breakdown,
    draw_fairness,
    draw_profile_breakdown,
    PROTO_COLORS,
    _proto_color,
)
from plots.handover import draw_handover_impact, draw_handover_schedule, draw_timeseries
from plots.timing import draw_timing_breakdown
