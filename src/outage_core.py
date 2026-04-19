"""
outage_core.py
--------------

Main outage definitions used by the thesis simulation results.
"""

from __future__ import annotations

from params import Rp_min, r_BL


def pu_outage(Rp: float, rate_threshold: float = Rp_min) -> int:
    """
    PU outage indicator.

    outage = 1 if Rp < Rp_min, otherwise 0.
    """
    return 1 if float(Rp) < float(rate_threshold) else 0


def su_outage(rate_s: float, rate_threshold: float = r_BL) -> int:
    """
    SU target-rate outage indicator.

    outage = 1 if the SU rate is below the SVC base-layer threshold.
    """
    return 1 if float(rate_s) + 1e-12 < float(rate_threshold) else 0


def solution_outage(
    Rp: float,
    Rsc: float,
    Reff_s: float,
    layers_s: int,
    pu_rate_threshold: float = Rp_min,
    su_base_layer_threshold: float = r_BL,
) -> dict[str, int | bool | float]:
    """
    Outage event used by the main optimized simulation.

    The optimizer is designed around PU protection and SVC base-layer
    decodability, so KQ6 is measured from these same quantities.
    """
    pu_out = pu_outage(Rp, pu_rate_threshold)
    bl_decodable = (
        float(Rsc) + 1e-12 >= float(su_base_layer_threshold)
        and float(Reff_s) + 1e-12 >= float(su_base_layer_threshold)
        and int(layers_s) >= 1
    )
    return {
        "PU_outage": int(pu_out),
        "SU_outage": int(not bl_decodable),
        "bl_decodable": bool(bl_decodable),
        "su_rate_threshold": float(su_base_layer_threshold),
    }

