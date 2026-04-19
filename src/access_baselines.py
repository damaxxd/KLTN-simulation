"""
access_baselines.py
-------------------

Baseline access schemes for KQ2 QoE-vs-SNR comparison.

Schemes:
    - OMA: non-CR orthogonal PU/SU resources with fixed 50/50 split.
    - NOMA: non-CR non-orthogonal PU/SU transmission with one SU video stream.
    - CR_NOMA: NOMA with cognitive-radio PU protection constraint.

The proposed CR-RSMA solver stays in power_solver_grid.py / power_solver_sca.py.
This module only provides comparable baseline solvers for KQ2.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from params import (
    Pp_max,
    Ps_max,
    GRID_POINTS_PU,
    GRID_POINTS_SU,
    Rp_min,
    r_BL,
    svc_layers,
)
from rate_model import achievable_rate_from_sinr
from svc_abstraction import decodable_layers, layer_status
from quality_model import summarize_quality
from outage_core import pu_outage, su_outage


BaselineScheme = Literal["OMA", "NOMA", "CR_NOMA"]

OMA_RESOURCE_FRACTION: float = 0.5
BASELINE_VIDEO_EFFICIENCY: dict[str, float] = {
    "OMA": 0.78,
    "NOMA": 0.74,
    "CR_NOMA": 0.82,
}


def _svc_from_total_rate(Rs: float, scheme: BaselineScheme) -> dict[str, object]:
    """
    Map one SU video stream to SVC effective rate.

    Baseline OMA/NOMA do not split common/private streams, so the total SU
    rate supports the cumulative SVC layers directly after a delivery
    efficiency factor. This mirrors the reference base-code abstraction:
    baselines do not protect BL via an RSMA common stream, so their useful
    video rate is lower than their raw PHY rate.
    """
    efficiency = BASELINE_VIDEO_EFFICIENCY[scheme.upper()]
    reff_s = min(efficiency * float(Rs), float(np.sum(svc_layers)))
    layers_s = decodable_layers(reff_s)
    status_s = layer_status(reff_s)
    return {
        "Reff_s": reff_s,
        "layers_s": layers_s,
        "status_s": status_s,
    }


def compute_baseline_rates(
    scheme: BaselineScheme,
    gp: float,
    gs: float,
    Pp: float,
    Ps: float,
    sigma2: float,
) -> dict[str, float]:
    """
    Compute PU/SU rates for a baseline access scheme.

    OMA uses a fixed half-resource split and no cross-interference.
    NOMA and CR-NOMA share the same PHY model; only CR-NOMA is filtered by
    the solver through the PU protection constraint.
    """
    scheme = scheme.upper()

    if scheme == "OMA":
        sinr_p = gp * Pp / max(sigma2, 1e-12)
        sinr_s = gs * Ps / max(sigma2, 1e-12)
        Rp = OMA_RESOURCE_FRACTION * achievable_rate_from_sinr(sinr_p)
        Rs = OMA_RESOURCE_FRACTION * achievable_rate_from_sinr(sinr_s)
    elif scheme in {"NOMA", "CR_NOMA"}:
        sinr_p = gp * Pp / max(gs * Ps + sigma2, 1e-12)
        sinr_s = gs * Ps / max(gp * Pp + sigma2, 1e-12)
        Rp = achievable_rate_from_sinr(sinr_p)
        Rs = achievable_rate_from_sinr(sinr_s)
    else:
        raise ValueError(f"Unknown baseline scheme: {scheme}")

    return {
        "sinr_p": float(sinr_p),
        "sinr_s": float(sinr_s),
        "Rp": float(Rp),
        "Rs": float(Rs),
    }


def evaluate_baseline_power(
    scheme: BaselineScheme,
    gp: float,
    gs: float,
    sigma2: float,
    Pp: float,
    Ps: float,
) -> dict[str, object]:
    """Evaluate one baseline power vector through PHY/SVC/QoE/outage blocks."""
    rates = compute_baseline_rates(
        scheme=scheme,
        gp=gp,
        gs=gs,
        Pp=Pp,
        Ps=Ps,
        sigma2=sigma2,
    )
    svc = _svc_from_total_rate(rates["Rs"], scheme)
    quality = summarize_quality(rates["Rp"], svc["Reff_s"], svc["layers_s"])
    outage = {
        "PU_outage": pu_outage(rates["Rp"]),
        "SU_outage": su_outage(svc["Reff_s"]),
    }

    return {
        "scheme": scheme,
        "Pp": float(Pp),
        "Ps": float(Ps),
        "Psc": 0.0,
        "Psp": float(Ps),
        "Ps_total": float(Ps),
        "sinr_p": rates["sinr_p"],
        "sinr_s": rates["sinr_s"],
        "sinr_sc": rates["sinr_s"],
        "sinr_sp": 0.0,
        "Rp": rates["Rp"],
        "Rsc": 0.0,
        "Rsp": rates["Rs"],
        "Rs": rates["Rs"],
        "Reff_s": float(svc["Reff_s"]),
        "layers_s": int(svc["layers_s"]),
        "status_s": svc["status_s"],
        "PSNR_p": float(quality["PSNR_p"]),
        "PSNR_s": float(quality["PSNR_s"]),
        "QoE_p": float(quality["QoE_p"]),
        "QoE_s": float(quality["QoE_s"]),
        "QoE_sys": float(quality["QoE_sys"]),
        "PU_outage": int(outage["PU_outage"]),
        "SU_outage": int(outage["SU_outage"]),
        "tau": 0.0,
        "su_interference": 0.0,
    }


def solve_baseline_power_grid(
    scheme: BaselineScheme,
    gp: float,
    gs: float,
    sigma2: float,
    pu_power_max: float = Pp_max,
    su_power_max: float = Ps_max,
    rp_min: float = Rp_min,
) -> dict[str, object] | None:
    """
    Grid-search solver for OMA/NOMA/CR-NOMA KQ2 baselines.

    All baselines require the SU stream to support the SVC base layer. Only
    CR-NOMA additionally enforces PU protection. OMA and NOMA remain non-CR
    comparison baselines.
    """
    scheme = scheme.upper()
    if scheme not in {"OMA", "NOMA", "CR_NOMA"}:
        raise ValueError(f"Unknown baseline scheme: {scheme}")

    enforce_pu_protection = scheme == "CR_NOMA"
    best_result: dict[str, object] | None = None
    best_obj = -1e18

    Pp_grid = np.linspace(0.05, pu_power_max, GRID_POINTS_PU)
    Ps_grid = np.linspace(0.0, su_power_max, GRID_POINTS_SU)

    for Pp in Pp_grid:
        for Ps in Ps_grid:
            if Ps <= 1e-12:
                continue

            result = evaluate_baseline_power(
                scheme=scheme,
                gp=gp,
                gs=gs,
                sigma2=sigma2,
                Pp=Pp,
                Ps=Ps,
            )

            if enforce_pu_protection and result["Rp"] < rp_min:
                continue

            if result["Reff_s"] < r_BL:
                continue

            obj = float(result["QoE_sys"])
            if best_result is None or obj > best_obj + 1e-9:
                best_obj = obj
                best_result = result

    if best_result is not None:
        best_result["objective"] = float(best_obj)

    return best_result
