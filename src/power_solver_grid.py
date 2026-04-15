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
        QoE_sys = w_p * PSNR_p + w_s * PSNR_s
"""

from __future__ import annotations

import numpy as np

from params import (
    Pp_max, Ps_max,
    Rp_min,
    w_p, w_s,
    GRID_POINTS_PU, GRID_POINTS_SU,
    r_BL,
)
from rate_model import compute_all_rates
from svc_abstraction import summarize_svc_state
from quality_model import summarize_quality
from outage_model import summarize_outage


def solve_power_grid(
    gp: float,
    gs: float,
    sigma2: float,
    pu_power_max: float = Pp_max,
    su_power_max: float = Ps_max,
    rp_min: float = Rp_min,
) -> dict[str, float] | None:
    """
    Grid-search solver for the current CR-RSMA + SVC + QoE=PSNR setup.
    """
    best_result: dict[str, float] | None = None
    best_obj = -1e18

    Pp_grid = np.linspace(0.05, pu_power_max, GRID_POINTS_PU)
    Psc_grid = np.linspace(0.0, su_power_max, GRID_POINTS_SU)
    Psp_grid = np.linspace(0.0, su_power_max, GRID_POINTS_SU)

    for Pp in Pp_grid:
        for Psc in Psc_grid:
            for Psp in Psp_grid:

                # Constraint 1: SU total power budget
                if Psc + Psp > su_power_max + 1e-12:
                    continue

                # Skip silent SU
                if Psc + Psp <= 1e-12:
                    continue

                # -------------------------------------------------
                # PHY block
                # -------------------------------------------------
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

                # Constraint 2: PU protection
                if Rp < rp_min:
                    continue

                # -------------------------------------------------
                # NEW Constraint 3: common stream must support BL
                # -------------------------------------------------
                # This is critical for keeping the RSMA -> SVC mapping meaningful.
                if Rsc < r_BL:
                    continue

                # -------------------------------------------------
                # SVC abstraction block
                # -------------------------------------------------
                svc = summarize_svc_state(Rsc, Rsp)
                Reff_s = svc["Reff_s"]

                # -------------------------------------------------
                # Quality block
                # -------------------------------------------------
                quality = summarize_quality(Rp, Reff_s, svc["layers_s"])

                # -------------------------------------------------
                # Outage block
                # -------------------------------------------------
                outage = summarize_outage(Rp, quality["PSNR_s"])

                # -------------------------------------------------
                # Objective
                # -------------------------------------------------
                obj = quality["QoE_sys"]

                # Optional tie-break:
                # among near-equal objectives, prefer larger Rsc
                if best_result is None:
                    update = True
                else:
                    update = False
                    if obj > best_obj + 1e-9:
                        update = True
                    elif abs(obj - best_obj) <= 1e-9 and Rsc > best_result["Rsc"]:
                        update = True

                if update:
                    best_obj = obj
                    best_result = {
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

                        "objective": float(obj),
                    }

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