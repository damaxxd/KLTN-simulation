"""
channel.py
----------

Khối 2: Wireless channel model

Input:
    - snr_db
    - link distance
    - eta, P_ref from params

Output:
    - sigma2 from SNR
    - channel power gain |h|^2
    - optional pair of gains (gp, gs)

Model:
    Rayleigh fading + path loss
"""

from __future__ import annotations

import numpy as np

from params import RNG_SEED, eta, P_ref

rng = np.random.default_rng(RNG_SEED)


def sigma2_from_snr_db(snr_db: float, p_ref: float = P_ref) -> float:
    """
    Convert SNR in dB to noise power.

    Formula:
        SNR_linear = p_ref / sigma2
    so:
        sigma2 = p_ref / 10^(snr_db/10)

    Input:
        snr_db : SNR in dB
        p_ref  : reference signal power

    Output:
        sigma2 : noise power
    """
    snr_lin = 10.0 ** (snr_db / 10.0)
    return p_ref / snr_lin


def sample_complex_rayleigh() -> complex:
    """
    Sample one circularly-symmetric complex Gaussian variable:
        g ~ CN(0,1)

    Output:
        one complex sample g
    """
    g_real = rng.normal(0.0, 1.0 / np.sqrt(2.0))
    g_imag = rng.normal(0.0, 1.0 / np.sqrt(2.0))
    return g_real + 1j * g_imag


def sample_channel_gain(distance: float = 1.0, pathloss_exp: float = eta) -> float:
    """
    Sample channel power gain |h|^2 under Rayleigh fading + path loss.

    Model:
        h = g / sqrt(1 + d^eta)
        g ~ CN(0,1)

    Therefore:
        gain = |h|^2

    Input:
        distance     : transmitter-receiver separation
        pathloss_exp : eta

    Output:
        gain : nonnegative channel power gain
    """
    g = sample_complex_rayleigh()
    h = g / np.sqrt(1.0 + distance ** pathloss_exp)
    return float(np.abs(h) ** 2)


def sample_pu_su_gains(
    d_pu: float = 1.2,
    d_su: float = 1.0,
    pathloss_exp: float = eta,
) -> tuple[float, float]:
    """
    Sample a pair of gains for PU and SU.

    Input:
        d_pu, d_su   : distances of PU and SU to the BS
        pathloss_exp : eta

    Output:
        gp, gs
    """
    gp = sample_channel_gain(distance=d_pu, pathloss_exp=pathloss_exp)
    gs = sample_channel_gain(distance=d_su, pathloss_exp=pathloss_exp)
    return gp, gs


if __name__ == "__main__":
    snr_db_test = 10.0
    sigma2 = sigma2_from_snr_db(snr_db_test)
    gp, gs = sample_pu_su_gains(d_pu=1.2, d_su=1.0)

    print("=== CHANNEL TEST ===")
    print(f"SNR (dB)   : {snr_db_test}")
    print(f"sigma2     : {sigma2:.6f}")
    print(f"gp         : {gp:.6f}")
    print(f"gs         : {gs:.6f}")