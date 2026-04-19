"""
paper_reference_baselines.py
----------------------------

Adapted paper-method baselines for the KQ5 PSNR-vs-SNR comparison.

These methods execute the main throughput/rate objectives and protection
constraints from the cited papers under the thesis system assumptions. They are
not exact reproductions of the original papers, because the thesis model has
one PU, one SU, uplink CR-RSMA, and SVC video, while the papers use different
network dimensions such as UAV-ISAC trajectory or multi-SU CRN-NOMA.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.optimize import minimize

from params import (
    Pp_max,
    Ps_max,
    GRID_POINTS_PU,
    GRID_POINTS_SU,
    Rp_min,
    r_BL,
    svc_cum,
)
from rate_model import compute_all_rates
from access_baselines import compute_baseline_rates
from svc_abstraction import (
    decodable_layers,
    layer_status,
    rsma_effective_video_rate,
)
from quality_model import summarize_quality
from outage_core import solution_outage
from interference_protection import (
    interference_threshold_tau,
    su_interference_power,
    su_residual_interference_power,
    su_respects_pu_residual_interference_budget,
)


ReferenceScheme = Literal["Feng_RSMA_UAV", "He_CR_NOMA_MTCC"]

REFERENCE_SCHEMES: tuple[ReferenceScheme, ...] = (
    "Feng_RSMA_UAV",
    "He_CR_NOMA_MTCC",
)

SLSQP_MAXITER = 80
SLSQP_FTOL = 1e-8
CONSTRAINT_TOL = 1e-6
MAX_MULTI_STARTS = 6


def _chebyshev_lobatto_nodes(lower: float, upper: float, count: int) -> np.ndarray:
    """Return deterministic Chebyshev-Lobatto nodes over a bounded interval."""
    if count <= 1:
        return np.array([(float(lower) + float(upper)) / 2.0])

    k = np.arange(count, dtype=float)
    nodes = 0.5 * (lower + upper) + 0.5 * (upper - lower) * np.cos(
        np.pi * k / float(count - 1)
    )
    return np.unique(np.clip(np.sort(nodes), lower, upper))


def _top_multistart_points(
    candidates: list[np.ndarray],
    objective_value,
    is_feasible,
    max_starts: int = MAX_MULTI_STARTS,
) -> list[np.ndarray]:
    """
    Select the strongest feasible Chebyshev/grid points for SLSQP starts.
    """
    scored: list[tuple[float, np.ndarray]] = []
    for x0 in candidates:
        if not is_feasible(x0):
            continue
        scored.append((float(objective_value(x0)), np.asarray(x0, dtype=float)))

    scored.sort(key=lambda item: item[0], reverse=True)
    starts: list[np.ndarray] = []
    for _, point in scored:
        if any(np.linalg.norm(point - existing) <= 1e-9 for existing in starts):
            continue
        starts.append(point)
        if len(starts) >= max_starts:
            break
    return starts


def _svc_from_rsma_reference(Rsc: float, Rsp: float) -> dict[str, object]:
    """
    Evaluate a throughput-driven RSMA allocation with SVC BL gating.

    The proposed solver enforces Rsc >= r_BL before SVC decoding. The reference
    throughput baseline deliberately does not optimize that constraint, so KQ5
    evaluation must prevent private-stream EL rate from creating decodable
    layers when the common-stream BL is missing.
    """
    if float(Rsc) < r_BL:
        return {
            "Reff_s": min(float(Rsc), r_BL),
            "layers_s": 0,
            "status_s": np.zeros(4, dtype=int),
        }

    reff_s = rsma_effective_video_rate(Rsc, Rsp)
    layers_s = decodable_layers(reff_s)
    return {
        "Reff_s": float(reff_s),
        "layers_s": int(layers_s),
        "status_s": layer_status(reff_s),
    }


def _svc_from_single_stream(Rs: float) -> dict[str, object]:
    """
    Evaluate a throughput-driven single SU stream through the SVC layer ladder.
    """
    reff_s = min(float(Rs), float(svc_cum[-1]))
    layers_s = decodable_layers(reff_s)
    return {
        "Reff_s": float(reff_s),
        "layers_s": int(layers_s),
        "status_s": layer_status(reff_s),
    }


def _with_quality_and_outage(
    result: dict[str, object],
    *,
    gp: float,
    gs: float,
    sigma2: float,
    common_interference: bool,
) -> dict[str, object]:
    quality = summarize_quality(
        float(result["Rp"]),
        float(result["Reff_s"]),
        int(result["layers_s"]),
    )
    outage = solution_outage(
        Rp=float(result["Rp"]),
        Rsc=float(result["Rsc"]),
        Reff_s=float(result["Reff_s"]),
        layers_s=int(result["layers_s"]),
    )
    tau = interference_threshold_tau(gp, float(result["Pp"]), sigma2)
    if common_interference:
        su_interference = su_interference_power(
            gs,
            float(result["Psc"]),
            float(result["Psp"]),
        )
    else:
        su_interference = su_residual_interference_power(gs, float(result["Psp"]))

    result.update(
        {
            "PSNR_p": float(quality["PSNR_p"]),
            "PSNR_s": float(quality["PSNR_s"]),
            "QoE_p": float(quality["QoE_p"]),
            "QoE_s": float(quality["QoE_s"]),
            "QoE_sys": float(quality["QoE_sys"]),
            "PU_outage": int(outage["PU_outage"]),
            "SU_outage": int(outage["SU_outage"]),
            "tau": float(tau),
            "su_interference": float(su_interference),
            "feasible": True,
        }
    )
    return result


def _solve_feng_rsma_uav_throughput(
    gp: float,
    gs: float,
    sigma2: float,
    pu_power_max: float,
    su_power_max: float,
    rp_min: float,
) -> dict[str, object] | None:
    """
    Feng et al. [2026]-adapted RSMA-UAV throughput method.

    The trajectory and sensing terms are disabled because the thesis model has
    no multi-slot UAV-ISAC state. The adapted method optimizes transmit powers
    for communication throughput under PU protection and power constraints.
    """
    best_result: dict[str, object] | None = None
    best_obj = -1e18

    def rates_from_x(x: np.ndarray) -> dict[str, float]:
        Pp, Psc, Psp = np.asarray(x, dtype=float)
        return compute_all_rates(
            gp=gp,
            gs=gs,
            Pp=float(Pp),
            Psc=float(Psc),
            Psp=float(Psp),
            sigma2=sigma2,
        )

    def throughput_value(x: np.ndarray) -> float:
        phy = rates_from_x(x)
        return float(phy["Rp"] + phy["Rsc"] + phy["Rsp"])

    def residual_interference_margin(x: np.ndarray) -> float:
        Pp, _, Psp = np.asarray(x, dtype=float)
        tau = interference_threshold_tau(gp, float(Pp), sigma2, rp_min)
        return float(tau - float(gs) * float(Psp))

    def is_feasible(x: np.ndarray) -> bool:
        Pp, Psc, Psp = np.asarray(x, dtype=float)
        if Pp < 0.05 - CONSTRAINT_TOL or Pp > pu_power_max + CONSTRAINT_TOL:
            return False
        if Psc < -CONSTRAINT_TOL or Psp < -CONSTRAINT_TOL:
            return False
        if Psc + Psp > su_power_max + CONSTRAINT_TOL:
            return False
        if Psc + Psp <= 1e-12:
            return False
        if rates_from_x(x)["Rp"] < rp_min - CONSTRAINT_TOL:
            return False
        return residual_interference_margin(x) >= -CONSTRAINT_TOL

    def build_result(x: np.ndarray, optimization_success: bool, status: str) -> dict[str, object]:
        Pp, Psc, Psp = np.asarray(x, dtype=float)
        phy = rates_from_x(x)
        svc = _svc_from_rsma_reference(phy["Rsc"], phy["Rsp"])
        obj = throughput_value(x)
        result = {
            "scheme": "Feng_RSMA_UAV",
            "method": "Feng et al. [2026] adapted SLSQP",
            "Pp": float(Pp),
            "Psc": float(Psc),
            "Psp": float(Psp),
            "Ps_total": float(Psc + Psp),
            "sinr_p": float(phy["sinr_p"]),
            "sinr_sc": float(phy["sinr_sc"]),
            "sinr_sp": float(phy["sinr_sp"]),
            "Rp": float(phy["Rp"]),
            "Rsc": float(phy["Rsc"]),
            "Rsp": float(phy["Rsp"]),
            "Rs": float(phy["Rs"]),
            "Reff_s": float(svc["Reff_s"]),
            "layers_s": int(svc["layers_s"]),
            "status_s": svc["status_s"],
            "objective": obj,
            "optimization_success": bool(optimization_success),
            "solver_status": status,
        }
        return _with_quality_and_outage(
            result,
            gp=gp,
            gs=gs,
            sigma2=sigma2,
            common_interference=False,
        )

    ppu_nodes = _chebyshev_lobatto_nodes(0.05, pu_power_max, min(GRID_POINTS_PU, 5))
    ps_nodes = _chebyshev_lobatto_nodes(0.0, su_power_max, min(GRID_POINTS_SU, 5))
    candidates = [
        np.array([Pp, Psc, Psp], dtype=float)
        for Pp in ppu_nodes
        for Psc in ps_nodes
        for Psp in ps_nodes
        if Psc + Psp <= su_power_max + 1e-12
    ]
    starts = _top_multistart_points(candidates, throughput_value, is_feasible)
    if not starts:
        return None

    constraints = [
        {"type": "ineq", "fun": lambda x: su_power_max - float(x[1]) - float(x[2])},
        {"type": "ineq", "fun": lambda x: rates_from_x(x)["Rp"] - rp_min},
        {"type": "ineq", "fun": residual_interference_margin},
    ]
    bounds = [(0.05, pu_power_max), (0.0, su_power_max), (0.0, su_power_max)]

    for x0 in starts:
        opt = minimize(
            fun=lambda x: -throughput_value(x),
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": SLSQP_MAXITER, "ftol": SLSQP_FTOL, "disp": False},
        )
        candidate_x = np.asarray(opt.x if opt.x is not None else x0, dtype=float)
        candidate_x = np.clip(candidate_x, [0.05, 0.0, 0.0], [pu_power_max, su_power_max, su_power_max])
        for point, success, status in (
            (x0, False, "chebyshev_start"),
            (candidate_x, bool(opt.success), str(opt.message)),
        ):
            if not is_feasible(point):
                continue
            candidate = build_result(point, success, status)
            obj = float(candidate["objective"])
            if best_result is None or obj > best_obj + 1e-9:
                best_result = candidate
                best_obj = obj

    return best_result


def _solve_he_cr_noma_mtcc_throughput(
    gp: float,
    gs: float,
    sigma2: float,
    pu_power_max: float,
    su_power_max: float,
    rp_min: float,
) -> dict[str, object] | None:
    """
    He et al. [2022]-adapted CR-NOMA/MTCC throughput method.

    The multi-SU admission stage degenerates to one SU in the thesis model. The
    adapted method keeps CR-NOMA throughput maximization, PU QoS, and the
    interference-temperature constraint.
    """
    best_result: dict[str, object] | None = None
    best_obj = -1e18

    def rates_from_x(x: np.ndarray) -> dict[str, float]:
        Pp, Ps = np.asarray(x, dtype=float)
        return compute_baseline_rates(
            scheme="CR_NOMA",
            gp=gp,
            gs=gs,
            Pp=float(Pp),
            Ps=float(Ps),
            sigma2=sigma2,
        )

    def throughput_value(x: np.ndarray) -> float:
        return float(rates_from_x(x)["Rs"])

    def interference_margin(x: np.ndarray) -> float:
        Pp, Ps = np.asarray(x, dtype=float)
        tau = interference_threshold_tau(gp, float(Pp), sigma2, rp_min)
        return float(tau - float(gs) * float(Ps))

    def is_feasible(x: np.ndarray) -> bool:
        Pp, Ps = np.asarray(x, dtype=float)
        if Pp < 0.05 - CONSTRAINT_TOL or Pp > pu_power_max + CONSTRAINT_TOL:
            return False
        if Ps <= 1e-12 or Ps > su_power_max + CONSTRAINT_TOL:
            return False
        if rates_from_x(x)["Rp"] < rp_min - CONSTRAINT_TOL:
            return False
        return interference_margin(x) >= -CONSTRAINT_TOL

    def build_result(x: np.ndarray, optimization_success: bool, status: str) -> dict[str, object]:
        Pp, Ps = np.asarray(x, dtype=float)
        rates = rates_from_x(x)
        svc = _svc_from_single_stream(rates["Rs"])
        obj = throughput_value(x)
        result = {
            "scheme": "He_CR_NOMA_MTCC",
            "method": "He et al. [2022] adapted SLSQP/MTCC",
            "Pp": float(Pp),
            "Psc": 0.0,
            "Psp": float(Ps),
            "Ps_total": float(Ps),
            "sinr_p": float(rates["sinr_p"]),
            "sinr_sc": float(rates["sinr_s"]),
            "sinr_sp": 0.0,
            "Rp": float(rates["Rp"]),
            "Rsc": 0.0,
            "Rsp": float(rates["Rs"]),
            "Rs": float(rates["Rs"]),
            "Reff_s": float(svc["Reff_s"]),
            "layers_s": int(svc["layers_s"]),
            "status_s": svc["status_s"],
            "objective": obj,
            "optimization_success": bool(optimization_success),
            "solver_status": status,
        }
        return _with_quality_and_outage(
            result,
            gp=gp,
            gs=gs,
            sigma2=sigma2,
            common_interference=True,
        )

    ppu_nodes = _chebyshev_lobatto_nodes(0.05, pu_power_max, min(GRID_POINTS_PU, 5))
    ps_nodes = _chebyshev_lobatto_nodes(0.0, su_power_max, min(GRID_POINTS_SU, 6))
    candidates = [
        np.array([Pp, Ps], dtype=float)
        for Pp in ppu_nodes
        for Ps in ps_nodes
    ]
    starts = _top_multistart_points(candidates, throughput_value, is_feasible)
    if not starts:
        return None

    constraints = [
        {"type": "ineq", "fun": lambda x: rates_from_x(x)["Rp"] - rp_min},
        {"type": "ineq", "fun": interference_margin},
    ]
    bounds = [(0.05, pu_power_max), (0.0, su_power_max)]

    for x0 in starts:
        opt = minimize(
            fun=lambda x: -throughput_value(x),
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": SLSQP_MAXITER, "ftol": SLSQP_FTOL, "disp": False},
        )
        candidate_x = np.asarray(opt.x if opt.x is not None else x0, dtype=float)
        candidate_x = np.clip(candidate_x, [0.05, 0.0], [pu_power_max, su_power_max])
        for point, success, status in (
            (x0, False, "chebyshev_start"),
            (candidate_x, bool(opt.success), str(opt.message)),
        ):
            if not is_feasible(point):
                continue
            candidate = build_result(point, success, status)
            obj = float(candidate["objective"])
            if best_result is None or obj > best_obj + 1e-9:
                best_result = candidate
                best_obj = obj

    return best_result


def solve_reference_power_grid(
    scheme: ReferenceScheme,
    gp: float,
    gs: float,
    sigma2: float,
    pu_power_max: float = Pp_max,
    su_power_max: float = Ps_max,
    rp_min: float = Rp_min,
) -> dict[str, object] | None:
    """Solve one KQ5 adapted paper-method reference baseline."""
    if scheme == "Feng_RSMA_UAV":
        return _solve_feng_rsma_uav_throughput(
            gp=gp,
            gs=gs,
            sigma2=sigma2,
            pu_power_max=pu_power_max,
            su_power_max=su_power_max,
            rp_min=rp_min,
        )
    if scheme == "He_CR_NOMA_MTCC":
        return _solve_he_cr_noma_mtcc_throughput(
            gp=gp,
            gs=gs,
            sigma2=sigma2,
            pu_power_max=pu_power_max,
            su_power_max=su_power_max,
            rp_min=rp_min,
        )
    raise ValueError(f"Unknown KQ5 reference scheme: {scheme}")
