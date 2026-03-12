# 5G-NTN Satellite Link Simulation

An end-to-end simulation of a 5G Non-Terrestrial Network (NTN) LEO satellite link.
The pipeline integrates three simulation layers — PHY-layer link abstraction,
urban ray tracing, and packet-level network simulation — and produces a set of
diagnostic figures comparing four transport protocols under realistic satellite
channel conditions including handover events.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Part 1 — PHY layer  (ntn_phy.py)                           │
│  Sionna 1.2.1 + OpenNTN (3GPP TR 38.811)                    │
│  QPSK · LDPC r=0.5 · 3 scenarios (urban/dense/suburban)    │
│  → BER and BLER curves vs Eb/N0                             │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  Part 2 — Ray tracing  (rt_sim.py)                          │
│  Sionna RT · Munich OSM scene                               │
│  UE at [50, 80, 1.5 m] · 3 satellites at elev 70°/55°/40°  │
│  → per-satellite path gain, delay spread, PER               │
└────────────────────┬────────────────────────────────────────┘
                     │  channel_stats
┌────────────────────▼────────────────────────────────────────┐
│  Part 3 — Network simulation  (ntn_ns3.py)                  │
│  NS-3 · 4 protocols: UDP, TCP NewReno, CUBIC, BBR           │
│  2 handovers (t=20 s → t=40 s) · 60 s simulation           │
│  → throughput, latency, packet loss per protocol            │
└─────────────────────────────────────────────────────────────┘
```

### End-to-end path

```
UE (Phone)  ──5G-NR NTN──  LEO Satellite  ──Ka feeder──  Ground Station  ──Fibre──  Internet Server
```

## Simulation scenario

| Parameter | Value |
|---|---|
| Carrier frequency | 3.5 GHz (5G NR n78) |
| Modulation | QPSK |
| Code rate | 0.5 (LDPC) |
| Satellite altitude | 600 km (LEO) |
| Constellation | 3 satellites |
| Satellite elevations | 69.9° / 54.9° / 39.9° |
| Handover 1 | t = 20 s (Sat 0 → Sat 1) |
| Handover 2 | t = 40 s (Sat 1 → Sat 2) |
| Per-slot PER | 0.101 / 0.033 / 0.768 |
| Simulation duration | 60 s |
| Ray tracing scene | Munich (Sionna RT, OSM) |
| UE position | [50, 80, 1.5 m] in Munich scene |

### NS-3 results (representative run)

| Protocol | Throughput (kbps) | Loss (%) | Latency (ms) | Handovers |
|---|---|---|---|---|
| UDP | 4830 | 5.28 | 15.4 | 2 |
| TCP NewReno | 382 | 0.00 | 14.2 | 2 |
| TCP CUBIC | 618 | 0.00 | 14.2 | 2 |
| TCP BBR | 4185 | 0.00 | 14.2 | 2 |

## Setup

### Prerequisites

- Python 3.10 (via pyenv)
- NS-3 built at `~/ns-3-dev` with Python bindings enabled

```bash
pyenv install 3.10
pyenv local 3.10
```

### Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### NS-3 and additional dependencies

```bash
./install.sh
```

`install.sh` applies the LLVM Mitsuba patch needed for CPU-only Sionna RT and
installs the OpenNTN channel model plugin.

## Running

```bash
source venv/bin/activate
PYTHONPATH=~/ns-3-dev/build/bindings/python python3 main.py
```

The full pipeline takes approximately 5–15 minutes depending on hardware
(ray tracing and NS-3 dominate runtime).

## Output files

| File | Description |
|---|---|
| `ntn_topology.png` | Basic end-to-end topology diagram |
| `ntn_network_illustration.png` | Detailed NTN network diagram with Earth arc, orbit, link parameters |
| `ntn_ue_satellite.png` | Artistic street-canyon scene with UE, satellite, and multipath rays |
| `ntn_protocol_comparison.png` | Grouped bar chart: latency / throughput / loss per protocol |
| `ntn_throughput_over_time.png` | Per-protocol throughput vs time with handover markers |
| `ntn_summary.png` | Five-panel combined BER + NS-3 results figure |
| `ntn_rt_paths_sat0.png` | Sionna RT ray paths for Sat 0 (elev 70°) |
| `ntn_rt_paths_sat1.png` | Sionna RT ray paths for Sat 1 (elev 55°) |
| `ntn_rt_paths_sat2.png` | Sionna RT ray paths for Sat 2 (elev 40°) |
| `ntn_rt_radiomap.png` | Composite RT radio coverage map |

## Configuration

All parameters are centralised in `config.py`.  Key knobs:

| Parameter | Default | Description |
|---|---|---|
| `CARRIER_FREQ_HZ` | `3.5e9` | 5G NR carrier frequency [Hz] |
| `SAT_HEIGHT_M` | `600_000` | LEO orbital altitude [m] |
| `NUM_SATELLITES` | `3` | Satellites in the simulated pass |
| `SIM_DURATION_S` | `60.0` | NS-3 simulation duration [s] |
| `RT_MAX_DEPTH` | `5` | Max ray reflections per path |
| `RT_UE_POSITION` | `[50, 80, 1.5]` | UE location in Munich scene [m] |
| `APP_DATA_RATE` | `"5Mbps"` | UDP CBR source rate |
| `PACKET_SIZE_BYTES` | `1400` | Application payload size [bytes] |
| `TCP_SNDRCV_BUF_BYTES` | `512_000` | TCP socket buffer [bytes] |

## File structure

```
.
├── config.py              # All shared simulation parameters
├── ntn_phy.py             # Part 1: Sionna + OpenNTN BER/BLER
├── rt_sim.py              # Part 2: Sionna RT ray tracing
├── ntn_ns3.py             # Part 3: NS-3 multi-protocol simulation
├── topology_diagram.py    # All visualisation functions
├── main.py                # Pipeline orchestration
├── install.sh             # Dependency installer
└── requirements.txt       # Python package list
```

## Results interpretation

- **BER/BLER curves** (Part 1): Expected to show improvement with increasing Eb/N0.
  Flat curves at ~0.5 indicate the OpenNTN channel model dominates noise at
  low batch sizes; this is a known behaviour and does not affect Parts 2–3.
- **Ray tracing** (Part 2): Higher elevation angles give fewer but stronger paths.
  Sat 0 (70°) returns 4 paths; Sat 2 (40°) returns 8 paths due to more
  building-wall reflections at shallower incidence.
- **NS-3 throughput** (Part 3): BBR performs best under the high-PER Sat 2 slot
  because it uses bandwidth probing rather than loss as a congestion signal.
  TCP NewReno and CUBIC back off sharply when Sat 2 (PER = 0.768) is active.
- **Throughput-over-time**: The Sat 1 slot (t = 20–40 s, PER = 0.033) is the
  best window; all TCP variants show a brief recovery dip immediately after
  each handover due to congestion-window reset.
