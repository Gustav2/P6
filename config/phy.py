"""
config/phy.py — PHY / OFDM simulation parameters
"""

CARRIER_FREQ_HZ = 2.0e9
"""
5G NR carrier frequency [Hz].
- 2.0 GHz (n255, S-band) is the 3GPP reference service-link frequency for
  LEO NTN as specified in 3GPP TR 38.821 §6.1, Table 6.1.1-1.
- n78 (3.5 GHz) is a terrestrial FR1 band not allocated for satellite
  service links; n255 (1980–2010 MHz UL / 2170–2200 MHz DL) is the 3GPP
  NTN S-band reference.
- At 2.0 GHz the FSPL over 550 km is ~4.9 dB lower than at 3.5 GHz,
  improving the link margin significantly.
- Affects OFDM wavelength, free-space path loss (FSPL), and Doppler shifts.
Source: [3GPP-38.821] §6.1, Table 6.1.1-1 (service-link reference = 2 GHz).
         [3GPP-38.101-5] §5.2 (NR band n255 definition).
"""

SUBCARRIER_SPACING = 30e3
"""
OFDM subcarrier spacing [Hz].
- 30 kHz = NR numerology µ=1, recommended by 3GPP TR 38.821 for S-band NTN.
- At 2 GHz the peak satellite Doppler is:
    Δf_max = v/c × f = 7612 / 3×10⁸ × 2×10⁹ ≈ ±50.7 kHz
  µ=0 (15 kHz SCS) would be overwhelmed (ICI ≈ 3× SCS); µ=1 (30 kHz)
  keeps the worst-case Doppler to ~1.7× SCS, within UE pre-compensation
  range per 3GPP TS 38.211 §7.3.
- One NR slot at µ=1 has duration T_slot = 0.5 ms (14 OFDM symbols).
Source: [3GPP-38.821] §6.1.2 (µ=1 reference numerology for S-band NTN).
         [3GPP-38.300] §5.3.1, Table 4.1-1 in [3GPP-38.101-1].
"""

FFT_SIZE = 128
"""
OFDM FFT size (number of subcarriers including guard bands).
- Together with SUBCARRIER_SPACING determines the OFDM symbol bandwidth.
- Bandwidth = FFT_SIZE × SUBCARRIER_SPACING = 1.92 MHz at defaults.
"""

NUM_OFDM_SYMBOLS = 14
"""Number of OFDM symbols per slot (1 NR slot at µ=0).
Source: [3GPP-38.300] §5.3.1 — 14 symbols per slot for normal cyclic prefix.
"""

CP_LENGTH = 9
"""
Cyclic prefix length [samples].
- Must exceed the maximum multipath delay spread to avoid ISI.
- 9 samples × (1/15 kHz) ≈ 4.7 µs; sufficient for urban delay spreads
  of ~1 µs but marginal for long-delay NTN multipath.
"""

PILOT_SYMBOL_IDX = [2, 11]
"""
OFDM symbol indices used for pilot (channel estimation) symbols.
- Two pilots per slot following 3GPP NR DMRS Type 1, positions 2 and 11.
Source: [3GPP-38.211] §7.4.1.1 (DMRS for PDSCH/PUSCH).
"""

NUM_BITS_PER_SYMBOL = 2
"""
Bits per modulation symbol.
- 2 = QPSK (MCS index 1). Change to 4 (16-QAM) or 6 (64-QAM) for higher
  spectral efficiency.
- QPSK is the reference modulation for link-budget calculations in
  3GPP TR 38.821 at low-to-medium SNR.
Source: [3GPP-38.214] Table 5.1.3.1-2 (PDSCH MCS tables).
"""

CODERATE = 0.5
"""
LDPC channel code rate (k/n).
- 0.5 = half-rate; 50 % overhead for error correction.
- Corresponds to MCS index ~7 for QPSK in 3GPP NR.
Source: [3GPP-38.214] Table 5.1.3.1-2.
"""

BATCH_SIZE = 512
"""
Number of independent channel realisations evaluated per SNR point.
- Higher values reduce Monte-Carlo variance but increase memory/time.
- 512 keeps all 8 Jetson CPU cores busy via TF intra-op parallelism while
  staying well within the 61 GB unified memory budget (~400 MB peak).
- Original value was 64; 512 gives 8× better variance reduction per sweep,
  which is important for reliable BER→PER sigmoid fitting.
"""
