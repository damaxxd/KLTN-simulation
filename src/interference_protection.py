"""
interference_protection.py
--------------------------

PU QoS protection and interference-temperature helpers.
"""

from __future__ import annotations

import math

from params import B, c1, c2, Rp_min


def target_sinr_threshold(rate_threshold: float) -> float:
    """
    Invert the AMC rate formula to get a target SINR.

    R = c1 * B * log2(1 + SINR / c2)
    SINR_target = c2 * (2^(R/(c1*B)) - 1)
    """
    return float(c2 * (2.0 ** (float(rate_threshold) / (c1 * B)) - 1.0))


def pu_target_sinr_threshold(rate_threshold: float = Rp_min) -> float:
    """Return the PU target SINR for the minimum PU rate."""
    return target_sinr_threshold(rate_threshold)


def interference_threshold_tau(
    gp: float,
    Pp: float,
    sigma2: float,
    rate_threshold: float = Rp_min,
) -> float:
    """
    Maximum received SU interference power that preserves the PU target rate.
    """
    gamma_target = pu_target_sinr_threshold(rate_threshold)
    if gamma_target <= 0.0:
        return float("inf")
    return max(0.0, float(gp) * float(Pp) / gamma_target - float(sigma2))


def pu_outage_qos_protected(
    gp: float,
    Pp: float,
    sigma2: float,
    rate_threshold: float = Rp_min,
) -> int:
    """
    PU outage under the OMA/no-SU protected condition.
    """
    gamma_target = pu_target_sinr_threshold(rate_threshold)
    no_su_signal = float(gp) * float(Pp)
    no_su_required = gamma_target * float(sigma2)
    return 1 if no_su_signal + 1e-12 < no_su_required else 0


def su_interference_power(gs: float, Psc: float, Psp: float) -> float:
    """Received total SU interference power seen by the PU."""
    return float(gs) * (float(Psc) + float(Psp))


def su_residual_interference_power(gs: float, Psp: float) -> float:
    """Received residual SU private-stream interference after common SIC."""
    return float(gs) * float(Psp)


def su_respects_pu_interference_budget(
    gp: float,
    gs: float,
    Pp: float,
    Psc: float,
    Psp: float,
    sigma2: float,
    rate_threshold: float = Rp_min,
) -> bool:
    """Return True if total SU interference does not exceed tau."""
    tau = interference_threshold_tau(gp, Pp, sigma2, rate_threshold)
    return su_interference_power(gs, Psc, Psp) <= tau + 1e-12


def su_respects_pu_residual_interference_budget(
    gp: float,
    gs: float,
    Pp: float,
    Psp: float,
    sigma2: float,
    rate_threshold: float = Rp_min,
) -> bool:
    """Return True if residual SU private interference does not exceed tau."""
    tau = interference_threshold_tau(gp, Pp, sigma2, rate_threshold)
    return su_residual_interference_power(gs, Psp) <= tau + 1e-12


def achievable_rate_from_sinr_for_outage(sinr: float) -> float:
    """Local AMC-adjusted rate helper used by paper-inspired outage models."""
    sinr = max(float(sinr), 1e-12)
    return float(c1 * B * math.log2(1.0 + sinr / c2))

