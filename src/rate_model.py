"""
rate_model.py
-------------

Khối 3: PHY rate model

Input:
    - gp, gs, sigma2
    - Pp, Psc, Psp
    - B, c1, c2 from params

Output:
    - PU SINR
    - SU common SINR
    - SU private SINR
    - Rp, Rsc, Rsp, Rs

Notes:
    - Achievable rate formula follows the AMC-adjusted model
      used in the quality-driven NOMA video paper.
    - SINR structure is adapted to the CR-RSMA uplink abstraction
      of this thesis.
"""

from __future__ import annotations

import numpy as np

from params import B, c1, c2


def achievable_rate_from_sinr(
    sinr: float,
    bandwidth: float = B,
    rate_adjust: float = c1,
    snr_gap: float = c2,
) -> float:
    """
    Compute achievable rate from SINR.

    Formula:
        R = c1 * B * log2(1 + SINR / c2)

    Input:
        sinr        : signal-to-interference-plus-noise ratio
        bandwidth   : B
        rate_adjust : c1
        snr_gap     : c2

    Output:
        rate in bit/s
    """
    sinr = max(float(sinr), 1e-12)
    return rate_adjust * bandwidth * np.log2(1.0 + sinr / snr_gap)


def sinr_pu(gp: float, gs: float, Pp: float, Psc: float, Psp: float, sigma2: float) -> float:
    """
    PU SINR under the proposed CR-RSMA SIC order.

    Model:
        Decode order: SU common -> PU -> SU private
        SINR_p = gp * Pp / (gs * Psp + sigma2)

    Interpretation:
        The SU common stream has already been decoded and canceled. The PU is
        protected only against residual SU private-stream interference.
    """
    _ = Psc
    denom = gs * Psp + sigma2
    return gp * Pp / max(denom, 1e-12)


def sinr_su_common(gp: float, gs: float, Pp: float, Psc: float, Psp: float, sigma2: float) -> float:
    """
    SU common-stream SINR.

    Model:
        SINR_sc = gs * Psc / (gs * Psp + gp * Pp + sigma2)

    Interpretation:
        The common stream is decoded in the presence of:
        - SU private-stream interference
        - PU interference
        - noise
    """
    denom = gs * Psp + gp * Pp + sigma2
    return gs * Psc / max(denom, 1e-12)


def sinr_su_private(gp: float, gs: float, Pp: float, Psc: float, Psp: float, sigma2: float) -> float:
    """
    SU private-stream SINR under the proposed CR-RSMA SIC order.

    Model:
        Decode order: SU common -> PU -> SU private
        SINR_sp = gs * Psp / sigma2

    Interpretation:
        The SU private stream is decoded after both the common stream and PU
        stream have been decoded and canceled.
    """
    _ = (gp, Pp, Psc)
    denom = sigma2
    return gs * Psp / max(denom, 1e-12)


def rate_pu(gp: float, gs: float, Pp: float, Psc: float, Psp: float, sigma2: float) -> float:
    """Return PU achievable rate."""
    gamma_p = sinr_pu(gp, gs, Pp, Psc, Psp, sigma2)
    return achievable_rate_from_sinr(gamma_p)


def rate_su_common(gp: float, gs: float, Pp: float, Psc: float, Psp: float, sigma2: float) -> float:
    """Return SU common-stream achievable rate."""
    gamma_sc = sinr_su_common(gp, gs, Pp, Psc, Psp, sigma2)
    return achievable_rate_from_sinr(gamma_sc)


def rate_su_private(gp: float, gs: float, Pp: float, Psc: float, Psp: float, sigma2: float) -> float:
    """Return SU private-stream achievable rate."""
    gamma_sp = sinr_su_private(gp, gs, Pp, Psc, Psp, sigma2)
    return achievable_rate_from_sinr(gamma_sp)


def compute_all_rates(
    gp: float,
    gs: float,
    Pp: float,
    Psc: float,
    Psp: float,
    sigma2: float,
) -> dict[str, float]:
    """
    Compute all relevant PHY rates.

    Output keys:
        sinr_p, sinr_sc, sinr_sp,
        Rp, Rsc, Rsp, Rs
    """
    gamma_p = sinr_pu(gp, gs, Pp, Psc, Psp, sigma2)
    gamma_sc = sinr_su_common(gp, gs, Pp, Psc, Psp, sigma2)
    gamma_sp = sinr_su_private(gp, gs, Pp, Psc, Psp, sigma2)

    Rp = achievable_rate_from_sinr(gamma_p)
    Rsc = achievable_rate_from_sinr(gamma_sc)
    Rsp = achievable_rate_from_sinr(gamma_sp)
    Rs = Rsc + Rsp

    return {
        "sinr_p": gamma_p,
        "sinr_sc": gamma_sc,
        "sinr_sp": gamma_sp,
        "Rp": Rp,
        "Rsc": Rsc,
        "Rsp": Rsp,
        "Rs": Rs,
    }


if __name__ == "__main__":
    # quick unit-style test
    gp_test = 0.40
    gs_test = 0.55
    Pp_test = 0.20
    Psc_test = 0.10
    Psp_test = 0.15
    sigma2_test = 0.10

    out = compute_all_rates(
        gp=gp_test,
        gs=gs_test,
        Pp=Pp_test,
        Psc=Psc_test,
        Psp=Psp_test,
        sigma2=sigma2_test,
    )

    print("=== RATE MODEL TEST ===")
    for k, v in out.items():
        print(f"{k:8s}: {v:.6f}")
