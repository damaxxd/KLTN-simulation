"""
power_solver_grid.py
--------------------

Khối 7A: Grid-search power solver

Purpose:
    A sanity-check solver before switching to the thesis-final SCA solver.

Input:
    - gp, gs, sigma2
    - constraints from params
    - helper functions from rate / SVC / quality / outage blocks

Output:
    - best feasible power allocation and all associated metrics

Objective:
    maximize weighted system QoE:
        QoE_sys = w_p * QoE_p + w_s * QoE_s

    In the current thesis setting:
        QoE_p = PSNR_p
        QoE_s = PSNR_s
"""

from __future__ import annotations

import numpy as np

from params import (
    Pp_max, Ps_max,
    Rp_min,
    GRID_POINTS_PU, GRID_POINTS_SU,
    GRID_REFINEMENT_ENABLED,
    GRID_REFINE_POINTS_PU,
    GRID_REFINE_POINTS_SU,
    r_BL,
)
from rate_model import compute_all_rates
from svc_abstraction import summarize_svc_state
from quality_model import summarize_quality
from outage_core import solution_outage
from interference_protection import (
    interference_threshold_tau,
    su_residual_interference_power,
    su_respects_pu_residual_interference_budget,
)


def solve_power_grid(
    gp: float,
    gs: float,
    sigma2: float,
    pu_power_max: float = Pp_max,
    su_power_max: float = Ps_max,
    rp_min: float = Rp_min,
) -> dict[str, float] | None:
    """
    Grid-search solver for the CR-RSMA + SVC + PSNR-QoE setup.
    """
    best_result: dict[str, float] | None = None
    best_obj = -1e18

    def should_update(
        candidate: dict[str, float],
        candidate_obj: float,
        current: dict[str, float] | None,
        current_obj: float,
    ) -> bool:
        if current is None:
            return True
        if candidate_obj > current_obj + 1e-9:
            return True
        if abs(candidate_obj - current_obj) > 1e-9:
            return False
        if int(candidate["layers_s"]) > int(current["layers_s"]):
            return True
        if int(candidate["layers_s"]) < int(current["layers_s"]):
            return False
        if candidate["Reff_s"] > current["Reff_s"] + 1e-9:
            return True
        if abs(candidate["Reff_s"] - current["Reff_s"]) > 1e-9:
            return False
        return candidate["Ps_total"] < current["Ps_total"] - 1e-9

    def evaluate_candidate(
        Pp: float,
        Psc: float,
        Psp: float,
    ) -> tuple[dict[str, float], float] | None:

        # Constraint 1: SU total power budget
        if Psc + Psp > su_power_max + 1e-12:
            return None

        # Skip silent SU
        if Psc + Psp <= 1e-12:
            return None

        # Paper-inspired PU protection after CR-RSMA common SIC:
        # only residual SU private interference must not exceed tau.
        if not su_respects_pu_residual_interference_budget(
            gp=gp,
            gs=gs,
            Pp=Pp,
            Psp=Psp,
            sigma2=sigma2,
            rate_threshold=rp_min,
        ):
            return None

        phy = compute_all_rates(
            gp=gp,
            gs=gs,
            Pp=Pp,
            Psc=Psc,
            Psp=Psp,
            sigma2=sigma2,
        )

        Rp = phy["Rp"]
        Rsc = phy["Rsc"]
        Rsp = phy["Rsp"]

        if Rp < rp_min or Rsc < r_BL:
            return None

        svc = summarize_svc_state(Rsc, Rsp)
        Reff_s = svc["Reff_s"]
        quality = summarize_quality(Rp, Reff_s, svc["layers_s"])
        outage = solution_outage(
            Rp=Rp,
            Rsc=Rsc,
            Reff_s=Reff_s,
            layers_s=svc["layers_s"],
        )
        tau = interference_threshold_tau(gp, Pp, sigma2)
        su_interference = su_residual_interference_power(gs, Psp)
        obj = float(quality["QoE_sys"])
        result = {
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
            "su_respects_tau": bool(su_interference <= tau + 1e-12),

            "objective": obj,
        }
        return result, obj

    Pp_grid = np.linspace(0.05, pu_power_max, GRID_POINTS_PU)
    Psc_grid = np.linspace(0.0, su_power_max, GRID_POINTS_SU)
    Psp_grid = np.linspace(0.0, su_power_max, GRID_POINTS_SU)

    for Pp in Pp_grid:
        for Psc in Psc_grid:
            for Psp in Psp_grid:
                evaluated = evaluate_candidate(float(Pp), float(Psc), float(Psp))
                if evaluated is None:
                    continue
                candidate, obj = evaluated
                if should_update(candidate, obj, best_result, best_obj):
                    best_result = candidate
                    best_obj = obj

    if best_result is not None and GRID_REFINEMENT_ENABLED:
        ppu_step = (
            (pu_power_max - 0.05) / max(GRID_POINTS_PU - 1, 1)
        )
        ps_step = su_power_max / max(GRID_POINTS_SU - 1, 1)
        Pp_refine = np.linspace(
            max(0.05, best_result["Pp"] - ppu_step),
            min(pu_power_max, best_result["Pp"] + ppu_step),
            GRID_REFINE_POINTS_PU,
        )
        Psc_refine = np.linspace(
            max(0.0, best_result["Psc"] - ps_step),
            min(su_power_max, best_result["Psc"] + ps_step),
            GRID_REFINE_POINTS_SU,
        )
        Psp_refine = np.linspace(
            max(0.0, best_result["Psp"] - ps_step),
            min(su_power_max, best_result["Psp"] + ps_step),
            GRID_REFINE_POINTS_SU,
        )
        for Pp in Pp_refine:
            for Psc in Psc_refine:
                for Psp in Psp_refine:
                    evaluated = evaluate_candidate(float(Pp), float(Psc), float(Psp))
                    if evaluated is None:
                        continue
                    candidate, obj = evaluated
                    if should_update(candidate, obj, best_result, best_obj):
                        best_result = candidate
                        best_obj = obj

    return best_result


if __name__ == "__main__":
    gp_test = 0.40
    gs_test = 0.55
    sigma2_test = 0.10

    sol = solve_power_grid(gp=gp_test, gs=gs_test, sigma2=sigma2_test)

    print("=== GRID SOLVER TEST ===")
    if sol is None:
        print("No feasible solution found.")
    else:
        for k, v in sol.items():
            print(f"{k:10s}: {v}")
