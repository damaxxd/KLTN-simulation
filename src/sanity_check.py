"""
sanity_check.py
---------------

Khối 9: End-to-end sanity check

Purpose:
    Run one single-SNR / single-channel-realization test
    through the current pipeline.

Input:
    - params
    - channel model
    - grid or SCA solver

Output:
    Printed end-to-end result for quick debugging
"""

from __future__ import annotations

from params import print_parameter_summary, USE_SCA_SOLVER
from channel import sigma2_from_snr_db, sample_pu_su_gains
from power_solver_grid import solve_power_grid
from power_solver_sca import solve_power_sca


def run_sanity_check(snr_db: float = 10.0, d_pu: float = 1.2, d_su: float = 1.0) -> None:
    print_parameter_summary()

    sigma2 = sigma2_from_snr_db(snr_db)
    gp, gs = sample_pu_su_gains(d_pu=d_pu, d_su=d_su)

    print("\n=== SANITY CHECK INPUT ===")
    print(f"SNR_dB      : {snr_db}")
    print(f"sigma2      : {sigma2:.6f}")
    print(f"gp          : {gp:.6f}")
    print(f"gs          : {gs:.6f}")

    if USE_SCA_SOLVER:
        sol = solve_power_sca(gp=gp, gs=gs, sigma2=sigma2)
        solver_name = "SCA"
    else:
        sol = solve_power_grid(gp=gp, gs=gs, sigma2=sigma2)
        solver_name = "GRID"

    print(f"\n=== SANITY CHECK OUTPUT ({solver_name}) ===")
    if sol is None:
        print("No feasible solution found.")
        return

    print("--- Optimal powers ---")
    print(f"Pp*         : {sol['Pp']:.6f} W")
    print(f"Psc*        : {sol['Psc']:.6f} W")
    print(f"Psp*        : {sol['Psp']:.6f} W")
    print(f"Ps_total*   : {sol['Ps_total']:.6f} W")

    print("\n--- SINRs ---")
    print(f"SINR_p      : {sol['sinr_p']:.6f}")
    print(f"SINR_sc     : {sol['sinr_sc']:.6f}")
    print(f"SINR_sp     : {sol['sinr_sp']:.6f}")

    print("\n--- Rates (bps) ---")
    print(f"Rp          : {sol['Rp']:.2f}")
    print(f"Rsc         : {sol['Rsc']:.2f}")
    print(f"Rsp         : {sol['Rsp']:.2f}")
    print(f"Rs          : {sol['Rs']:.2f}")
    print(f"Reff_s      : {sol['Reff_s']:.2f}")

    print("\n--- SVC ---")
    print(f"layers_s    : {sol['layers_s']}")
    print(f"status_s    : {sol['status_s']}")

    print("\n--- Quality ---")
    print(f"PSNR_p      : {sol['PSNR_p']:.4f} dB")
    print(f"PSNR_s      : {sol['PSNR_s']:.4f} dB")
    print(f"QoE_p       : {sol['QoE_p']:.4f}")
    print(f"QoE_s       : {sol['QoE_s']:.4f}")
    print(f"QoE_sys     : {sol['QoE_sys']:.4f}")

    print("\n--- Outage ---")
    print(f"PU_outage   : {sol['PU_outage']}")
    print(f"SU_outage   : {sol['SU_outage']}")

    if "history" in sol:
        print("\n--- SCA history ---")
        print(f"iterations  : {len(sol['history'])}")
        if len(sol["history"]) > 0:
            print(f"last QoE_sys: {sol['history'][-1]['QoE_sys']:.4f}")


if __name__ == "__main__":
    run_sanity_check()