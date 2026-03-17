"""
draw_diagram.py — Standalone architecture diagram of the NTN simulation.

Run with:
    python3 draw_diagram.py
Output: diagram.png
"""

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Colour palette ─────────────────────────────────────────────────────────────
C_BG      = "#0d1117"
C_PANEL   = "#161b22"
C_TEXT    = "#e6edf3"
C_MUTED   = "#8b949e"
C_PHONE   = "#3fb950"
C_SAT     = "#d2a8ff"
C_GS      = "#f0883e"
C_SERVER  = "#58a6ff"
C_ISL     = "#f0883e"
C_SVC     = "#3fb950"
C_FEEDER  = "#58a6ff"
C_FIBRE   = "#58a6ff"
C_HO      = "#ff7b72"

fig_w, fig_h = 22, 14
fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=C_BG)
ax.set_facecolor(C_BG)
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")

# ── Drawing helpers ────────────────────────────────────────────────────────────

def node(cx, cy, w, h, border_color, title, sub1=None, sub2=None,
         title_size=9.5, sub_size=7.5):
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.08",
        linewidth=1.4, edgecolor=border_color, facecolor=C_PANEL, zorder=3)
    ax.add_patch(rect)
    lines = [l for l in [title, sub1, sub2] if l]
    total = len(lines)
    step = 0.28 if total > 1 else 0
    y0 = cy + step * (total - 1) / 2
    for k, line in enumerate(lines):
        yy = y0 - k * step
        sz = title_size if k == 0 else sub_size
        col = border_color if k == 0 else C_MUTED
        fw = "bold" if k == 0 else "normal"
        ax.text(cx, yy, line, ha="center", va="center",
                fontsize=sz, color=col, fontweight=fw, zorder=4)


def link(x0, y0, x1, y1, color, lw=1.6, dashed=False,
         mid_label=None, label_dx=0.0, label_dy=0.25, label_size=7.5,
         rad=0.0):
    ls = (0, (5, 3)) if dashed else "solid"
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}"),
        zorder=2)
    if mid_label:
        mx = (x0 + x1) / 2 + label_dx
        my = (y0 + y1) / 2 + label_dy
        ax.text(mx, my, mid_label, ha="center", va="center",
                fontsize=label_size, color=C_MUTED, zorder=5)


def divider(y):
    ax.plot([1.2, fig_w - 0.3], [y, y],
            color=C_MUTED, lw=0.5, linestyle="--", alpha=0.4, zorder=1)


def layer_tag(y, text):
    ax.text(0.2, y, text, ha="left", va="center",
            fontsize=8, color=C_MUTED, style="italic",
            rotation=90, zorder=5)


# ==============================================================================
#  Y coordinates for each layer
# ==============================================================================
Y_SAT    = 11.8   # satellites
Y_GS     = 8.2    # ground station
Y_SRV    = 5.5    # internet server
Y_PHONE  = 2.3    # phones

# Horizontal dividers
for y in [3.6, 6.9, 10.0]:
    divider(y)

# Layer labels (rotated, left margin)
layer_tag((Y_SAT + 10.0) / 2,  "SPACE  (LEO 550 km)")
layer_tag((Y_GS  +  6.9) / 2,  "GROUND SEGMENT")
layer_tag((Y_SRV +  3.6) / 2,  "INTERNET")
layer_tag((Y_PHONE + 0.1) / 2, "USER DEVICES")

# ==============================================================================
#  Satellite nodes
#  Benchmark Sat (Sat 0) — leftmost, always connected to GS
#  Access Sat 1, Access Sat 2 — right side, connected via ISL
# ==============================================================================
BW, BH = 3.0, 1.1

SAT0_X = 7.0
SAT1_X = 13.0
SAT2_X = 18.5

node(SAT0_X, Y_SAT, BW, BH, C_SAT,
     "Benchmark Sat  (Sat 0)",
     "elev ≈ 70°  |  one-way delay ≈ 1.9 ms",
     "PER ≈ 0.004  (EIRP = 23 dBm, QPSK r=0.5)")

node(SAT1_X, Y_SAT, BW, BH, C_SAT,
     "Access Sat 1",
     "elev ≈ 55°  |  delay ≈ 2.2 ms  |  PER ≈ 0.002",
     "serves t = 0 → 13.8 s")

node(SAT2_X, Y_SAT, BW, BH, C_SAT,
     "Access Sat 2",
     "elev ≈ 40°  |  delay ≈ 2.7 ms  |  PER ≈ 0.009",
     "serves t = 13.8 → 60 s")

# ISL: Access Sat 1 → Benchmark Sat
link(SAT1_X - BW/2, Y_SAT,
     SAT0_X + BW/2, Y_SAT,
     color=C_ISL, lw=2.0,
     mid_label="ISL   1 Gbps  |  delay 5 ms  |  PER 10⁻⁴",
     label_dy=0.3)

# ISL: Access Sat 2 → Access Sat 1 (then relays to Benchmark)
link(SAT2_X - BW/2, Y_SAT,
     SAT1_X + BW/2, Y_SAT,
     color=C_ISL, lw=2.0,
     mid_label="ISL   1 Gbps  |  delay 5 ms  |  PER 10⁻⁴",
     label_dy=0.3)

# Handover annotations below satellites
ax.text((SAT1_X + SAT2_X) / 2, Y_SAT - 0.85,
        "handover at t = 13.8 s                    handover at t = 45.6 s",
        ha="center", va="center", fontsize=8, color=C_HO,
        style="italic")
ax.annotate("", xy=(SAT1_X - 0.2, Y_SAT - 0.72),
            xytext=(SAT2_X - 1.5, Y_SAT - 0.72),
            arrowprops=dict(arrowstyle="<->", color=C_HO, lw=1.0))

# ==============================================================================
#  Ground Station
# ==============================================================================
node(SAT0_X, Y_GS, BW, BH, C_GS,
     "Ground Station (GS)",
     "Ka-band dish  42.7 dBi  |  26.5 GHz",
     "feeder link delay ≈ 2.3 ms")

# Feeder link: Benchmark Sat → GS  (vertical)
link(SAT0_X, Y_SAT - BH/2,
     SAT0_X, Y_GS  + BH/2,
     color=C_FEEDER, lw=2.2,
     mid_label="Ka-band feeder  3500 Mbps  |  PER ≈ 0.000",
     label_dx=2.2, label_dy=0.0)

# ==============================================================================
#  Internet Server
# ==============================================================================
node(SAT0_X, Y_SRV, BW, BH, C_SERVER,
     "Internet Server",
     "BulkSend (TCP) / OnOff (UDP)",
     "one shared server for all clients")

# GS → Server
link(SAT0_X, Y_GS  - BH/2,
     SAT0_X, Y_SRV + BH/2,
     color=C_FIBRE, lw=2.2,
     mid_label="Terrestrial fibre   1 Gbps  |  10 ms one-way",
     label_dx=2.4, label_dy=0.0)

# ==============================================================================
#  Client phones  (5 total)
# ==============================================================================
phone_xs = [3.2, 5.5, 7.8, 10.5, 13.2]
phone_info = [
    ("Phone 1", "stationary", "ConstantPosition"),
    ("Phone 2", "stationary", "ConstantPosition"),
    ("Phone 3", "stationary", "ConstantPosition"),
    ("Phone 4", "moving",     "RandomWaypoint  1–5 m/s"),
    ("Phone 5", "moving",     "RandomWaypoint  1–5 m/s"),
]

PW, PH = 2.05, 1.0
for px, (name, kind, mob) in zip(phone_xs, phone_info):
    col = C_PHONE
    node(px, Y_PHONE, PW, PH, col, name, kind, mob,
         title_size=9, sub_size=7)

# 500 m radius brace under phones
brace_y = Y_PHONE - 0.72
ax.annotate("", xy=(phone_xs[-1] + PW/2 + 0.1, brace_y),
            xytext=(phone_xs[0] - PW/2 - 0.1, brace_y),
            arrowprops=dict(arrowstyle="<->", color=C_MUTED, lw=0.9))
ax.text(sum(phone_xs) / len(phone_xs), brace_y - 0.28,
        "All 5 clients placed within 500 m radius  →  "
        "identical satellite visibility, same handover schedule & PER for all",
        ha="center", va="center", fontsize=7.5, color=C_MUTED)

# ==============================================================================
#  Service links: each phone → Access Sat 1 (initial serving satellite)
#  We fan the lines to a junction point, then a single annotated arrow up
# ==============================================================================
JX, JY = SAT1_X, 5.0   # junction / convergence point

# Thin lines from each phone to junction
for px in phone_xs:
    ax.plot([px, JX], [Y_PHONE + PH/2, JY],
            color=C_SVC, lw=0.9, linestyle="-", alpha=0.6, zorder=2)

# Junction marker
ax.plot(JX, JY, "o", color=C_SVC, markersize=6, zorder=4)

# Thick arrow from junction to Access Sat 1
link(JX, JY,
     SAT1_X, Y_SAT - BH/2,
     color=C_SVC, lw=2.4,
     mid_label="5G-NR NTN service link\n10 Mbps  |  PER = f(elevation, EIRP=23 dBm)\none-way delay = 2.2–2.7 ms",
     label_dx=2.6, label_dy=0.0)

# Junction label
ax.text(JX + 0.15, JY - 0.3, "shared uplink\n(5 clients)", ha="left",
        va="top", fontsize=7, color=C_SVC, zorder=5)

# ==============================================================================
#  Right panel: protocol legend + key parameters
# ==============================================================================
PNL_X = 19.5   # left edge of right info panel

# ── Protocol legend ────────────────────────────────────────────────────────────
ax.text(PNL_X, 13.2, "Protocols compared",
        ha="left", fontsize=9.5, color=C_TEXT, fontweight="bold")

protocols = [
    ("UDP",         "#8b949e", "CBR  1 Mbps/client  (5 Mbps total)"),
    ("TCP NewReno", "#58a6ff", "BulkSend · SACK · 10 MB cap"),
    ("TCP CUBIC",   "#3fb950", "BulkSend · SACK · 10 MB cap"),
    ("TCP BBR",     "#ffa657", "BulkSend · SACK · 10 MB cap"),
    ("QUIC",        "#d2a8ff", "UDP + RFC 9000 analytic corrections"),
]
for j, (name, col, desc) in enumerate(protocols):
    yy = 12.75 - j * 0.58
    ax.plot([PNL_X], [yy + 0.06], "s", color=col, markersize=7, zorder=5)
    ax.text(PNL_X + 0.35, yy + 0.06, name,
            ha="left", va="center", fontsize=8.5, color=col, fontweight="bold")
    ax.text(PNL_X + 0.35, yy - 0.18, desc,
            ha="left", va="center", fontsize=7, color=C_MUTED)

# ── Key parameters ─────────────────────────────────────────────────────────────
ax.text(PNL_X, 9.55, "Key parameters",
        ha="left", fontsize=9.5, color=C_TEXT, fontweight="bold")

params = [
    ("Orbit altitude",      "550 km  (Starlink Shell 1)"),
    ("Carrier frequency",   "3.5 GHz  (5G NR  n78)"),
    ("Satellite speed",     "7612 m/s  |  constellation: 3 sats"),
    ("Clients",             "3 stationary + 2 moving"),
    ("Mobility model",      "RandomWaypoint  1–5 m/s  within 500 m"),
    ("Phone EIRP",          "23 dBm  (UE Power Class 3)"),
    ("Sim duration",        "60 s  |  2 handovers  (t=13.8 s, t=45.6 s)"),
    ("Service link rate",   "10 Mbps  (bottleneck)"),
    ("ISL rate",            "1 Gbps  |  5 ms  |  PER 10⁻⁴"),
    ("Feeder link",         "Ka-band 26.5 GHz  |  ~3500 Mbps"),
    ("Backhaul",            "1 Gbps fibre  |  10 ms"),
    ("Channel model",       "Sionna RT path gains → PER sigmoid"),
    ("QUIC",                "Emulated: BBR + RFC 9000/9002 deltas"),
]
for j, (k, v) in enumerate(params):
    yy = 9.05 - j * 0.50
    ax.text(PNL_X, yy, k + ":", ha="left", va="center",
            fontsize=7.5, color=C_MUTED)
    ax.text(PNL_X + 2.05, yy, v, ha="left", va="center",
            fontsize=7.5, color=C_TEXT)

# ── Simulation pipeline ────────────────────────────────────────────────────────
ax.text(PNL_X, 2.55, "Simulation pipeline",
        ha="left", fontsize=9.5, color=C_TEXT, fontweight="bold")

pipeline = [
    ("#58a6ff", "Part 1", "Sionna + OpenNTN  →  BER/BLER curve  →  PER sigmoid"),
    ("#d2a8ff", "Part 2", "Sionna RT (Munich)  →  per-satellite path gains"),
    ("#3fb950", "Part 3", "NS-3  →  5 protocols × 60 s  →  FlowMonitor"),
]
for j, (col, tag, desc) in enumerate(pipeline):
    yy = 2.05 - j * 0.50
    ax.text(PNL_X, yy, tag, ha="left", va="center",
            fontsize=8, color=col, fontweight="bold")
    ax.text(PNL_X + 0.72, yy, desc, ha="left", va="center",
            fontsize=7.5, color=C_MUTED)

# ==============================================================================
#  Title
# ==============================================================================
ax.text(fig_w / 2, fig_h - 0.35,
        "5G-NTN ISL-Relay Satellite Simulation — Architecture",
        ha="center", va="center", fontsize=14, color=C_TEXT, fontweight="bold")
ax.text(fig_w / 2, fig_h - 0.75,
        "Direct topology  (USE_BASE_STATIONS = False)  ·  "
        "Sionna 1.2.1 + Sionna RT + NS-3  ·  3GPP TR 38.821 / TR 38.811",
        ha="center", va="center", fontsize=8.5, color=C_MUTED)

# ==============================================================================
#  Save
# ==============================================================================
plt.tight_layout(pad=0)
plt.savefig("diagram.png", dpi=160, bbox_inches="tight",
            facecolor=C_BG, edgecolor="none")
print("Saved diagram.png")
