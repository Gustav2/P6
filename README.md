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
| Carrier frequency | 2.0 GHz (5G NR n255, S-band NTN) |
| Numerology | µ=1 · 30 kHz SCS |
| Modulation / code rate | QPSK · LDPC r=0.5 |
| Satellite altitude | 550 km (LEO, Starlink Shell 1) |
| Visible satellites simulated | 12 |
| Protocols compared | UDP, TCP NewReno, TCP CUBIC, TCP BBR, QUIC |
| Traffic profiles | streaming, gaming, texting, voice, bulk |
| Clients | 30 stationary + 20 moving (pedestrian + vehicular) |
| Simulation duration | 300 s |
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
| `RT_MAX_DEPTH` | `5` | Max ray reflections per path |
| `SIM_DURATION_S` | `300.0` | NS-3 simulation duration [s] |

## Model Scope

- NS-3 uses an impaired-link abstraction (P2P + error models + handover scheduler),
  not a full 5G NR NTN stack.
- QUIC is reported as `QUIC (analytical)` and derived from a TCP BBR baseline
  with RFC 9000/9002-inspired corrections.
- Transport comparisons default to matched offered load
  (`TRANSPORT_COMPARE_MODE=matched_offered_load`) so protocol deltas are not
  confounded by traffic-generator mismatch.
- PHY defaults to realism mode (`PHY_REALISM_MODE=realistic`) and injects
  residual Doppler uncertainty (`RESIDUAL_DOPPLER_HZ_STD`).

## Reproducibility

- Use `SIM_SEED` and `SIM_RUN` to reproduce/stride random streams.
- Example:

```bash
SIM_SEED=42 SIM_RUN=1 TRANSPORT_COMPARE_MODE=matched_offered_load \
PYTHONPATH=~/ns-3-dev/build/bindings/python python3 main.py
```
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
