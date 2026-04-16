# 5G-NTN Satellite Link Simulation

An end-to-end simulation of a 5G Non-Terrestrial Network (NTN) LEO satellite link.
The pipeline integrates three simulation layers — PHY-layer link abstraction,
urban ray tracing, and packet-level network simulation — and produces a set of
diagnostic figures comparing multiple transport protocols under realistic satellite
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
│  Multi-UE samples in Munich OSM scene                        │
│  → per-satellite path gain percentiles, delay spread, PER    │
└────────────────────┬────────────────────────────────────────┘
                     │  channel_stats
┌────────────────────▼────────────────────────────────────────┐
│  Part 3 — Network simulation  (ntn_ns3.py)                  │
│  NS-3 · UDP, TCP NewReno/CUBIC/BBR, QUIC                    │
│  realistic moving+stationary traffic mix · 60 s simulation  │
│  → throughput, latency, packet loss per protocol            │
└─────────────────────────────────────────────────────────────┘
```

### End-to-end path

```
UE (Phone)  ──5G-NR NTN──  LEO Satellite  ──direct link──  Internet Server
```

No ISL or ground-station hops — the satellite connects directly to the
internet server node. The service link (phone→satellite) is the sole
bottleneck and the only link with a realistic error model.

## Simulation scenario

| Parameter | Value |
|---|---|
| Carrier frequency | 3.5 GHz (5G NR n78) |
| Modulation | QPSK |
| Code rate | 0.5 (LDPC) |
| Satellite altitude | 550 km (LEO) |
| Constellation context | ~3000 satellites total shell |
| Sampled visible satellites | 8 per simulated pass |
| Traffic profiles | streaming, gaming, texting, voice (+ bulk background) |
| Mobility classes | stationary, pedestrian, vehicular modem |
| Simulation duration | 60 s |
| Ray tracing scene | Munich (Sionna RT, OSM) |
| RT UE sampling | multiple UE points in Munich scene |

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
| `ntn_protocol_comparison.png` | Grouped bar chart: latency / throughput / loss per protocol |
| `ntn_summary.png` | Five-panel combined BER + NS-3 results figure |
| `ntn_latency_breakdown.png` | Per-hop latency breakdown (NTN + protocol overhead) |
| `ntn_handover_impact.png` | Per-slot throughput showing handover impact per protocol |
| `ntn_timeseries.png` | Per-protocol throughput vs time with handover markers |
| `ntn_fairness.png` | Jain's fairness index per protocol |
| `ntn_profile_breakdown.png` | Per-traffic-profile throughput breakdown |
| `ntn_results.png` | BER/BLER curves from PHY simulation |
| `ntn_rt_paths_sat*.png` | Sionna RT ray paths per satellite |
| `ntn_rt_radiomap.png` | Composite RT radio coverage map |

## Configuration

All parameters are centralised in `config.py`.  Key knobs:

| Parameter | Default | Description |
|---|---|---|
| `CARRIER_FREQ_HZ` | `3.5e9` | 5G NR carrier frequency [Hz] |
| `SAT_HEIGHT_M` | `550_000` | LEO orbital altitude [m] |
| `VISIBLE_SATELLITES_PER_PASS` | `8` | Sampled visible satellites per pass |
| `SIM_DURATION_S` | `60.0` | NS-3 simulation duration [s] |
| `RT_MAX_DEPTH` | `5` | Max ray reflections per path |
| `RT_RENDER_PATHS` | `True` | Set `False` to skip path renders and save RT time |
| `RT_RENDER_NUM_SAMPLES` | `64` | Path-tracing samples per pixel for RT renders |
| `RT_GAIN_P10_BLEND` | `0.35` | Blend between mean and p10 RT gain in PER model |
| `NUM_PEDESTRIAN_MOVING_CLIENTS` | `10` | Pedestrian moving clients |
| `NUM_VEHICULAR_MOVING_CLIENTS` | `10` | Vehicular-modem moving clients |
| `PEDESTRIAN_SPEED_MIN_MS / MAX` | `1.0 / 2.2` | Pedestrian speed range [m/s] |
| `VEHICULAR_SPEED_MIN_MS / MAX` | `10.0 / 22.0` | Vehicular speed range [m/s] |
| `APP_DATA_RATE` | `"5Mbps"` | UDP CBR source rate |
| `PACKET_SIZE_BYTES` | `1400` | Application payload size [bytes] |
| `TCP_SNDRCV_BUF_BYTES` | `512_000` | TCP socket buffer [bytes] |
| `SAT_SERVER_DATARATE` | `"1Gbps"` | Satellite → server link rate (never bottleneck) |

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
- **NS-3 throughput** (Part 3): BBR performs best under high-PER slots because
  it uses bandwidth probing rather than loss as a congestion signal.
  TCP NewReno and CUBIC back off sharply at high PER.
- **Throughput-over-time**: The highest-elevation satellite slot is the best
  window; all TCP variants show a brief recovery dip immediately after each
  handover due to congestion-window reset.
