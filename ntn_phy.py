"""
ntn_phy.py — Part 1: Sionna 1.2.1 + OpenNTN BER/BLER simulation
=================================================================
Simulates the NTN downlink physical layer using OpenNTN's TR38.811
channel models (DenseUrban, Urban, SubUrban) over an OFDM link with
QPSK modulation and LDPC channel coding.

Dependencies
------------
  Sionna 1.2.1        sionna.phy.*
  OpenNTN (main)      sionna.phy.channel.tr38811.*
                        (installed via install.sh into Sionna's channel dir)
"""

import numpy as np
import tensorflow as tf
import sionna
import sionna.phy
from sionna.phy import Block
from sionna.phy.ofdm import (
    ResourceGrid,
    ResourceGridMapper,
    LSChannelEstimator,
    LMMSEEqualizer,
    RemoveNulledSubcarriers,
)
from sionna.phy.channel import OFDMChannel
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.mimo import StreamManagement
from sionna.phy.utils import ebnodb2no, compute_ber
from sionna.phy.channel.tr38811 import AntennaArray
from sionna.phy.channel.tr38811 import DenseUrban, Urban, SubUrban

from config import (
    CARRIER_FREQ_HZ,
    SUBCARRIER_SPACING,
    FFT_SIZE,
    NUM_OFDM_SYMBOLS,
    CP_LENGTH,
    PILOT_SYMBOL_IDX,
    NUM_BITS_PER_SYMBOL,
    CODERATE,
    BATCH_SIZE,
    SAT_HEIGHT_M,
    ELEVATION_ANGLE_DEG,
)


# =============================================================================
# Channel model factory
# =============================================================================

def build_channel_model(scenario: str):
    """
    Construct an OpenNTN TR38.811 channel model for the given scenario.

    Parameters
    ----------
    scenario : str
        One of ``"dense_urban"``, ``"urban"``, ``"suburban"``.

    Returns
    -------
    An OpenNTN channel model instance (DenseUrban / Urban / SubUrban).
    """
    sat_array = AntennaArray(
        num_rows=1,
        num_cols=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=CARRIER_FREQ_HZ,
    )

    ue_array = AntennaArray(
        num_rows=1,
        num_cols=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=CARRIER_FREQ_HZ,
    )

    cls = {"dense_urban": DenseUrban, "urban": Urban, "suburban": SubUrban}[scenario]

    return cls(
        carrier_frequency=CARRIER_FREQ_HZ,
        ut_array=ue_array,
        bs_array=sat_array,
        direction="downlink",
        elevation_angle=ELEVATION_ANGLE_DEG,
        enable_pathloss=True,
        enable_shadow_fading=True,
    )


# =============================================================================
# End-to-end OFDM model
# =============================================================================

class NTNOFDMModel(Block):
    """
    Sionna 1.2.1 end-to-end NTN downlink model (SISO QPSK LDPC).

    Chain
    -----
    BinarySource → LDPC5GEncoder → Mapper → ResourceGridMapper
      → OFDMChannel (OpenNTN TR38.811)
      → LSChannelEstimator → LMMSEEqualizer
      → Demapper → LDPC5GDecoder
    """

    def __init__(self, channel_model):
        super().__init__()

        self._sm = StreamManagement(
            rx_tx_association=np.array([[1]]),
            num_streams_per_tx=1,
        )

        self._rg = ResourceGrid(
            num_ofdm_symbols=NUM_OFDM_SYMBOLS,
            fft_size=FFT_SIZE,
            subcarrier_spacing=SUBCARRIER_SPACING,
            num_tx=1,
            num_streams_per_tx=1,
            cyclic_prefix_length=CP_LENGTH,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=PILOT_SYMBOL_IDX,
        )

        self._n = int(self._rg.num_data_symbols * NUM_BITS_PER_SYMBOL)
        self._k = int(self._n * CODERATE)

        self._src    = BinarySource()
        self._enc    = LDPC5GEncoder(self._k, self._n)
        self._mapper = Mapper("qam", NUM_BITS_PER_SYMBOL)
        self._rg_map = ResourceGridMapper(self._rg)

        self._channel = OFDMChannel(
            channel_model=channel_model,
            resource_grid=self._rg,
            add_awgn=True,
            normalize_channel=True,
            return_channel=True,
        )

        self._rm_null = RemoveNulledSubcarriers(self._rg)
        self._ls_est  = LSChannelEstimator(self._rg, interpolation_type="nn")
        self._lmmse   = LMMSEEqualizer(self._rg, self._sm)
        self._demap   = Demapper("app", "qam", NUM_BITS_PER_SYMBOL)
        self._dec     = LDPC5GDecoder(self._enc, hard_out=True)

    @tf.function(jit_compile=False)
    def call(self, batch_size, ebno_db):
        no   = ebnodb2no(ebno_db, NUM_BITS_PER_SYMBOL, CODERATE, self._rg)

        b    = self._src([batch_size, 1, 1, self._k])
        cw   = self._enc(b)
        syms = self._mapper(cw)
        x    = self._rg_map(syms)

        y, _ = self._channel(x, no)

        h_hat, err_var = self._ls_est(y, no)
        x_hat, no_eff  = self._lmmse(y, h_hat, err_var, no)
        llr            = self._demap(x_hat, no_eff)
        b_hat          = self._dec(llr)

        return b, b_hat


# =============================================================================
# BER / BLER sweep
# =============================================================================

def run_sionna_ber(snr_db_range: np.ndarray, scenario: str):
    """
    Run a BER/BLER vs Eb/N0 sweep for the given NTN scenario.

    Parameters
    ----------
    snr_db_range : np.ndarray
        Array of Eb/N0 values in dB to evaluate.
    scenario : str
        ``"dense_urban"``, ``"urban"``, or ``"suburban"``.

    Returns
    -------
    ber_arr : np.ndarray
    bler_arr : np.ndarray
    """
    print(f"\n[Sionna + OpenNTN]  {scenario.upper()}"
          f"  LEO {SAT_HEIGHT_M/1e3:.0f} km  elev {ELEVATION_ANGLE_DEG:.0f}deg\n")

    sionna.phy.config.seed = 42

    ch_model = build_channel_model(scenario)

    ut_loc          = tf.zeros([BATCH_SIZE, 1, 3], dtype=tf.float32)
    bs_loc          = tf.tile(
        tf.constant([[[0., 0., SAT_HEIGHT_M]]], dtype=tf.float32),
        [BATCH_SIZE, 1, 1],
    )
    ut_orientations = tf.zeros([BATCH_SIZE, 1, 3], dtype=tf.float32)
    bs_orientations = tf.zeros([BATCH_SIZE, 1, 3], dtype=tf.float32)
    ut_velocities   = tf.tile(
        tf.constant([[[7600., 0., 0.]]], dtype=tf.float32),
        [BATCH_SIZE, 1, 1],
    )
    in_state        = tf.zeros([BATCH_SIZE, 1], dtype=tf.bool)

    ch_model.set_topology(
        ut_loc,
        bs_loc,
        ut_orientations,
        bs_orientations,
        ut_velocities,
        tf.cast(in_state, tf.bool),
    )

    model = NTNOFDMModel(ch_model)

    ber_arr, bler_arr = [], []
    for ebno_db in snr_db_range:
        b, b_hat = model(BATCH_SIZE, tf.cast(ebno_db, tf.float32))
        ber  = float(compute_ber(b, b_hat).numpy())
        bler = float(tf.reduce_mean(
            tf.cast(tf.reduce_any(tf.not_equal(b, b_hat), axis=-1),
                    tf.float32)).numpy())
        ber_arr.append(ber)
        bler_arr.append(bler)
        print(f"  Eb/N0 = {ebno_db:5.1f} dB   BER = {ber:.5f}   BLER = {bler:.5f}")

    return np.array(ber_arr), np.array(bler_arr)
