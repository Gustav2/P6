# 5G-NTN Satellite Link Simulation

End-to-end simulation of a 5G Non-Terrestrial Network (NTN) LEO satellite link,
integrating PHY-layer link abstraction, urban ray tracing, and packet-level
network simulation.

```
UE (Phone)  ──5G-NR NTN──  LEO Satellite  ──direct link──  Internet Server
```

## Pipeline

| Stage | Module | Tool | Output |
|---|---|---|---|
| 1 — PHY | `sim/phy.py` | Sionna 1.2.1 + OpenNTN | BER/BLER vs Eb/N0 |
| 2 — Ray tracing | `sim/ray_tracing.py` | Sionna RT · Munich OSM | per-satellite path gain, delay spread |
| 3 — Network | `sim/ns3.py` | NS-3 | throughput / latency / loss per protocol |

## Scenario

| Parameter | Value |
|---|---|
| Carrier frequency | 3.5 GHz (5G NR n78) |
| Modulation / code rate | QPSK · LDPC r=0.5 |
| Satellite altitude | 550 km (LEO, Starlink Shell 1) |
| Visible satellites simulated | 8 |
| Protocols compared | UDP, TCP NewReno, TCP CUBIC, TCP BBR, QUIC |
| Traffic profiles | streaming, gaming, texting, voice, bulk |
| Clients | 30 stationary + 20 moving (pedestrian + vehicular) |
| Simulation duration | 60 s |
| Ray tracing scene | Munich (Sionna RT, OpenStreetMap) |

## Setup

**Prerequisites:** Python 3.10, NS-3 at `~/ns-3-dev` with Python bindings.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./install.sh          # applies Mitsuba LLVM patch + installs OpenNTN
```

## Running

```bash
source venv/bin/activate
PYTHONPATH=~/ns-3-dev/build/bindings/python python3 main.py
```

Ray tracing and NS-3 dominate runtime (~5–10 min on a desktop GPU, longer on CPU-only).

## Output

All figures are written to `output/`.

| File | Description |
|---|---|
| `ntn_ber_bler.png` | BER and BLER vs Eb/N0 — all three OpenNTN scenarios |
| `ntn_protocol_comparison.png` | Grouped bars: latency / throughput / loss / jitter per protocol |
| `ntn_link_budget_waterfall.png` | Per-satellite link budget waterfall |
| `ntn_snr_vs_elevation.png` | SNR vs elevation with PER sigmoid overlay |
| `ntn_latency_breakdown.png` | Per-hop latency breakdown |
| `ntn_handover_impact.png` | Per-slot throughput showing handover impact |
| `ntn_handover_schedule.png` | Gantt timeline of the satellite handover schedule |
| `ntn_timeseries.png` | Per-second throughput with handover gap markers |
| `ntn_fairness.png` | Jain's fairness index per protocol |
| `ntn_profile_breakdown.png` | Throughput / loss by traffic profile |
| `ntn_channel_validation.png` | FSPL, Rician K-factor, delay spread vs 3GPP TR 38.811 |
| `ntn_timing.png` | Wall-clock runtime per pipeline stage |
| `ntn_rt_paths_sat*.png` | Sionna RT ray paths per satellite |
| `ntn_rt_radiomap.png` | Composite radio coverage map |

## Configuration

Parameters are split across the `config/` package:

| File | Contents |
|---|---|
| `config/phy.py` | Carrier frequency, OFDM numerology, LDPC code rate |
| `config/satellite.py` | Altitude, constellation size, handover thresholds |
| `config/ray_tracing.py` | RT scene settings, UE positions, render options |
| `config/network.py` | NS-3 duration, protocols, client counts, traffic profiles |

Key knobs:

| Parameter | Default | Description |
|---|---|---|
| `RT_RENDER_PATHS` | `True` | Set `False` to skip per-satellite renders and save time |
| `RT_MAX_DEPTH` | `5` | Max ray reflections per path |
| `SIM_DURATION_S` | `60.0` | NS-3 simulation duration [s] |
| `NUM_PEDESTRIAN_MOVING_CLIENTS` | `10` | Pedestrian moving clients |
| `NUM_VEHICULAR_MOVING_CLIENTS` | `10` | Vehicular moving clients |

## File structure

```
├── main.py              # Pipeline orchestration
├── config/              # Shared parameters (split by domain)
├── sim/
│   ├── phy.py           # Part 1: Sionna + OpenNTN BER/BLER
│   ├── ray_tracing.py   # Part 2: Sionna RT ray tracing
│   └── ns3.py           # Part 3: NS-3 multi-protocol simulation
├── plots/               # All visualisation functions (split by topic)
├── scripts/
│   └── draw_diagram.py  # Standalone architecture diagram
├── install.sh           # Dependency installer
└── requirements.txt
```
