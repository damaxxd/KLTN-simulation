"""
power_solver_sca.py
-------------------

Khối 7B: SCA-based power solver

Purpose:
    Thesis-facing solver using SCA (Successive Convex Approximation).
    A feasible initialization is first obtained from the grid solver.

Input:
    - gp, gs, sigma2
    - constraints and QoE settings from params

Output:
    - optimal power allocation
    - associated PHY / SVC / quality / outage metrics
    - SCA convergence history

Main idea:
    Each non-convex rate is written as a difference of two concave logs.
    The second concave term is linearized at the current iterate, yielding
    a concave lower bound surrogate. The surrogate problem is then solved
    iteratively.

Academic note:
    - The SCA principle follows the thesis direction for solving
      non-convex communication/video optimization problems.
    - The exact CR-RSMA uplink + SVC coupling is thesis-specific.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from params import (
    Pp_max, Ps_max, Rp_min, r_BL, svc_layers, svc_cum,
    w_p, w_s,
    SCA_MAX_ITER, SCA_TOL,
)
from rate_model import compute_all_rates
from svc_abstraction import summarize_svc_state
from quality_model import (
    summarize_quality,
    psnr_pu_from_rate,
    psnr_su_from_rate,
    qoe_from_psnr,
)
from outage_core import solution_outage
from interference_protection import (
    interference_threshold_tau,
    pu_target_sinr_threshold,
    su_residual_interference_power,
)
from power_solver_grid import solve_power_grid

try:
    from scipy.optimize import minimize
except ImportError as exc:
    raise ImportError(
        "SciPy is required for power_solver_sca.py. "
        "Please install it with: pip install scipy"
    ) from exc


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def _log2(x: float) -> float:
    """Safe log2."""
    return np.log2(max(float(x), 1e-12))


def _lin_upper_log2(z: float, z0: float) -> float:
    """
    First-order Taylor upper bound of concave log2(z) at z0:

        log2(z) <= log2(z0) + (z - z0)/(z0 ln 2)

    Why this matters:
    If a rate is written as:
        concave_part - concave_part
    then linearizing the second term by its upper bound yields
    a concave lower bound surrogate.

    Input:
        z  : current evaluation point
        z0 : expansion point

    Output:
        affine upper bound value of log2(z)
    """
    z = max(float(z), 1e-12)
    z0 = max(float(z0), 1e-12)
    return _log2(z0) + (z - z0) / (z0 * np.log(2.0))


def _surrogate_rates(
    x: np.ndarray,
    xk: np.ndarray,
    gp: float,
    gs: float,
    sigma2: float,
    c1B: float,
    c2_val: float,
) -> tuple[float, float, float]:
    """
    Build SCA lower-bound surrogate rates for:
        Rp, Rsc, Rsp

    Variables:
        x  = [Pp, Psc, Psp, tc, tp]
        xk = current iterate

    Output:
        Rp_lb, Rsc_lb, Rsp_lb
    """
    Pp, Psc, Psp, _, _ = x
    Pp_k, Psc_k, Psp_k, _, _ = xk

    # -----------------------------------------------------
    # PU rate after common-stream SIC:
    #
    # SINR_p = gp Pp / (gs Psp + sigma2)
    #
    # R_p = c1B * log2(1 + SINR_p/c2)
    #     = c1B * [ log2( gp Pp + c2*(gsPsp+sigma2) )
    #               - log2( c2*(gsPsp+sigma2) ) ]
    # -----------------------------------------------------
    _ = Psc
    Ap = gp * Pp + c2_val * (gs * Psp + sigma2)
    Bp = c2_val * (gs * Psp + sigma2)

    Bp_k = c2_val * (gs * Psp_k + sigma2)
    Rp_lb = c1B * (_log2(Ap) - _lin_upper_log2(Bp, Bp_k))

    # -----------------------------------------------------
    # SU common rate:
    #
    # SINR_sc = gs Psc / (gs Psp + gp Pp + sigma2)
    #
    # R_sc = c1B * [ log2( gs Psc + c2*(gsPsp + gpPp + sigma2) )
    #                - log2( c2*(gsPsp + gpPp + sigma2) ) ]
    # -----------------------------------------------------
    Asc = gs * Psc + c2_val * (gs * Psp + gp * Pp + sigma2)
    Bsc = c2_val * (gs * Psp + gp * Pp + sigma2)

    Bsc_k = c2_val * (gs * Psp_k + gp * Pp_k + sigma2)
    Rsc_lb = c1B * (_log2(Asc) - _lin_upper_log2(Bsc, Bsc_k))

    # -----------------------------------------------------
    # SU private rate after common-stream and PU SIC:
    #
    # SINR_sp = gs Psp / sigma2
    #
    # R_sp = c1B * [ log2( gs Psp + c2*sigma2 )
    #                - log2( c2*sigma2 ) ]
    # -----------------------------------------------------
    _ = Pp_k
    Asp = gs * Psp + c2_val * sigma2
    Bsp = c2_val * sigma2
    Rsp_lb = c1B * (_log2(Asp) - _log2(Bsp))

    return Rp_lb, Rsc_lb, Rsp_lb


def _approx_layers_from_reff(Reff_sur: float) -> int:
    """
    Approximate number of decodable layers from surrogate effective rate.

    Rule:
        compare Reff_sur against cumulative SVC thresholds

    Input:
        Reff_sur : surrogate effective rate tc + tp

    Output:
        integer layer count in {0,1,2,3,4}
    """
    return int(np.sum(Reff_sur >= svc_cum))


def _objective_neg(
    x: np.ndarray,
    xk: np.ndarray,
    gp: float,
    gs: float,
    sigma2: float,
    c1B: float,
    c2_val: float,
    r_el_total: float,
) -> float:
    """
    Negative surrogate objective for minimization.

    x = [Pp, Psc, Psp, tc, tp]

    Auxiliary variables:
        tc <= Rsc_lb, tc <= r_BL
        tp <= Rsp_lb, tp <= r_EL_total

    Surrogate effective rate:
        Reff_sur = tc + tp

    QoE:
        QoE_p = qoe_from_psnr(PSNR_p)
        QoE_s = qoe_from_psnr(PSNR_s)

    In the current thesis setting, qoe_from_psnr(PSNR) returns PSNR directly.

    Objective:
        maximize QoE_sys = w_p * QoE_p + w_s * QoE_s
    """
    Rp_lb, _, _ = _surrogate_rates(x, xk, gp, gs, sigma2, c1B, c2_val)
    _, _, _, tc, tp = x

    Rp_lb = max(Rp_lb, 1e-9)
    Reff_sur = max(tc + tp, 1e-9)

    QoE_p = qoe_from_psnr(psnr_pu_from_rate(Rp_lb))
    QoE_s = qoe_from_psnr(psnr_su_from_rate(Reff_sur))

    obj = w_p * QoE_p + w_s * QoE_s
    return -float(obj)


# =========================================================
# MAIN SOLVER
# =========================================================

def solve_power_sca(
    gp: float,
    gs: float,
    sigma2: float,
    max_iter: int = SCA_MAX_ITER,
    tol: float = SCA_TOL,
) -> Dict[str, Any] | None:
    """
    Solve the power allocation problem using SCA.

    Strategy:
        1) use grid solver to obtain a feasible initialization
        2) run SCA outer loop
        3) solve each surrogate subproblem with SLSQP

    Input:
        gp, gs   : channel gains
        sigma2   : noise power
        max_iter : SCA maximum number of iterations
        tol      : stopping tolerance

    Output:
        solution dictionary, or None if infeasible
    """
    from params import c1, B, c2

    c1B = c1 * B
    r_el_total = float(np.sum(svc_layers[1:]))
    gamma_target = pu_target_sinr_threshold(Rp_min)

    # -----------------------------------------------------
    # STEP 1: FEASIBLE INITIALIZATION FROM GRID SOLVER
    # -----------------------------------------------------
    init_sol = solve_power_grid(gp=gp, gs=gs, sigma2=sigma2)
    if init_sol is None:
        return None

    # x = [Pp, Psc, Psp, tc, tp]
    xk = np.array([
        init_sol["Pp"],
        init_sol["Psc"],
        init_sol["Psp"],
        min(init_sol["Rsc"], r_BL),
        min(init_sol["Rsp"], r_el_total),
    ], dtype=float)

    history: list[dict[str, float]] = []

    bounds = [
        (1e-6, Pp_max),      # Pp
        (0.0, Ps_max),       # Psc
        (0.0, Ps_max),       # Psp
        (0.0, r_BL),         # tc
        (0.0, r_el_total),   # tp
    ]

    for it in range(max_iter):

        # -------------------------
        # Constraints for surrogate problem
        # -------------------------
        def cons_su_power(x: np.ndarray) -> float:
            return Ps_max - x[1] - x[2]

        def cons_residual_interference(x: np.ndarray) -> float:
            return gp * x[0] / gamma_target - sigma2 - gs * x[2]

        def cons_pu_rate(x: np.ndarray) -> float:
            Rp_lb, _, _ = _surrogate_rates(x, xk, gp, gs, sigma2, c1B, c2)
            return Rp_lb - Rp_min

        def cons_bl_support(x: np.ndarray) -> float:
            _, Rsc_lb, _ = _surrogate_rates(x, xk, gp, gs, sigma2, c1B, c2)
            return Rsc_lb - r_BL

        def cons_tc_le_rsc(x: np.ndarray) -> float:
            _, Rsc_lb, _ = _surrogate_rates(x, xk, gp, gs, sigma2, c1B, c2)
            return Rsc_lb - x[3]

        def cons_tp_le_rsp(x: np.ndarray) -> float:
            _, _, Rsp_lb = _surrogate_rates(x, xk, gp, gs, sigma2, c1B, c2)
            return Rsp_lb - x[4]

        constraints = [
            {"type": "ineq", "fun": cons_su_power},
            {"type": "ineq", "fun": cons_residual_interference},
            {"type": "ineq", "fun": cons_pu_rate},
            {"type": "ineq", "fun": cons_bl_support},
            {"type": "ineq", "fun": cons_tc_le_rsc},
            {"type": "ineq", "fun": cons_tp_le_rsp},
        ]

        res = minimize(
            fun=_objective_neg,
            x0=xk,
            args=(xk, gp, gs, sigma2, c1B, c2, r_el_total),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-9, "disp": False},
        )

        # If surrogate solve fails, keep the current iterate and stop
        if not res.success:
            break

        x_new = res.x

        # -------------------------
        # Evaluate true model at new point
        # -------------------------
        Pp, Psc, Psp, tc, tp = x_new

        phy = compute_all_rates(
            gp=gp,
            gs=gs,
            Pp=Pp,
            Psc=Psc,
            Psp=Psp,
            sigma2=sigma2,
        )

        svc = summarize_svc_state(phy["Rsc"], phy["Rsp"])
        quality = summarize_quality(phy["Rp"], svc["Reff_s"], svc["layers_s"])

        history.append({
            "iter": float(it + 1),
            "Pp": float(Pp),
            "Psc": float(Psc),
            "Psp": float(Psp),
            "tc": float(tc),
            "tp": float(tp),
            "QoE_sys": float(quality["QoE_sys"]),
        })

        # stopping rule
        if np.linalg.norm(x_new - xk) <= tol:
            xk = x_new
            break

        xk = x_new

    # -----------------------------------------------------
    # FINAL TRUE-MODEL EVALUATION
    # -----------------------------------------------------
    Pp, Psc, Psp, tc, tp = xk

    phy = compute_all_rates(
        gp=gp,
        gs=gs,
        Pp=Pp,
        Psc=Psc,
        Psp=Psp,
        sigma2=sigma2,
    )

    # enforce true feasibility
    residual_ok = gs * Psp <= gp * Pp / gamma_target - sigma2 + 1e-12
    if phy["Rp"] < Rp_min or phy["Rsc"] < r_BL or not residual_ok:
        # fallback to initial feasible grid solution
        init_sol["history"] = history
        return init_sol

    svc = summarize_svc_state(phy["Rsc"], phy["Rsp"])
    quality = summarize_quality(phy["Rp"], svc["Reff_s"], svc["layers_s"])
    outage = solution_outage(
        Rp=phy["Rp"],
        Rsc=phy["Rsc"],
        Reff_s=svc["Reff_s"],
        layers_s=svc["layers_s"],
    )
    tau = interference_threshold_tau(gp, Pp, sigma2)
    su_interference = su_residual_interference_power(gs, Psp)

    sol: Dict[str, Any] = {
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

        "PSNR_p": float(quality["PSNR_p"]),
        "PSNR_s": float(quality["PSNR_s"]),
        "QoE_p": float(quality["QoE_p"]),
        "QoE_s": float(quality["QoE_s"]),
        "QoE_sys": float(quality["QoE_sys"]),

        "PU_outage": int(outage["PU_outage"]),
        "SU_outage": int(outage["SU_outage"]),
        "tau": float(tau),
        "su_interference": float(su_interference),

        "tc": float(tc),
        "tp": float(tp),
        "history": history,
    }
    return sol


if __name__ == "__main__":
    gp_test = 0.240660
    gs_test = 0.361960
    sigma2_test = 0.1

    sol = solve_power_sca(gp=gp_test, gs=gs_test, sigma2=sigma2_test)

    print("=== SCA SOLVER TEST ===")
    if sol is None:
        print("No feasible SCA solution found.")
    else:
        for k, v in sol.items():
            if k == "history":
                print(f"{k:10s}: {len(v)} iterations")
            else:
                print(f"{k:10s}: {v}")
