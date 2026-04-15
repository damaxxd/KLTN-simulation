"""
simulation_main.py
------------------

Khối 10: Full Monte Carlo simulation engine

Input:
    - params
    - channel
    - solver (grid or SCA)

Output:
    - aggregated simulation table
    - CSV file saved to disk

Important note:
    PU_Outage and SU_Outage are computed over ALL Monte Carlo realizations,
    not only over feasible solutions. This makes outage statistics meaningful.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from params import SNR_dB_list, N_MC, USE_SCA_SOLVER
from channel import sigma2_from_snr_db, sample_pu_su_gains
from power_solver_grid import solve_power_grid
from power_solver_sca import solve_power_sca


def run_single_realization(snr_db: float, d_pu: float = 1.2, d_su: float = 1.0) -> dict | None:
    """
    Run one Monte Carlo realization at one SNR point.

    Input:
        snr_db : current SNR in dB
        d_pu   : PU distance
        d_su   : SU distance

    Output:
        solution dict if feasible
        None otherwise
    """
    sigma2 = sigma2_from_snr_db(snr_db)
    gp, gs = sample_pu_su_gains(d_pu=d_pu, d_su=d_su)

    if USE_SCA_SOLVER:
        sol = solve_power_sca(gp=gp, gs=gs, sigma2=sigma2)
    else:
        sol = solve_power_grid(gp=gp, gs=gs, sigma2=sigma2)

    return sol


def aggregate_snr_point(snr_db: float, n_mc: int = N_MC) -> dict:
    """
    Aggregate all Monte Carlo realizations for one SNR point.

    VERY IMPORTANT:
    - Average metrics (power, rates, PSNR, QoE, layers) are computed
      over feasible solutions only.
    - Outage statistics are computed over ALL realizations.
    """
    feasible_records = []

    # outage counters over ALL realizations
    pu_outage_count = 0
    su_outage_count = 0
    infeasible_count = 0

    for _ in range(n_mc):
        sol = run_single_realization(snr_db)

        if sol is None:
            # If solver fails to find a feasible solution,
            # count it as outage/failure for both users.
            # This is the key fix that makes outage meaningful.
            pu_outage_count += 1
            su_outage_count += 1
            infeasible_count += 1
            continue

        feasible_records.append(sol)

        pu_outage_count += int(sol["PU_outage"])
        su_outage_count += int(sol["SU_outage"])

    feasible_count = len(feasible_records)

    if feasible_count == 0:
        return {
            "SNR_dB": snr_db,
            "Avg_Pp": np.nan,
            "Avg_Psc": np.nan,
            "Avg_Psp": np.nan,
            "Avg_Rp": np.nan,
            "Avg_Rsc": np.nan,
            "Avg_Rsp": np.nan,
            "Avg_Rs": np.nan,
            "Avg_Reff_s": np.nan,
            "Avg_Layers_s": np.nan,
            "Avg_PSNR_p": np.nan,
            "Avg_PSNR_s": np.nan,
            "Avg_QoE_p": np.nan,
            "Avg_QoE_s": np.nan,
            "Avg_QoE_sys": np.nan,
            "PU_Outage": pu_outage_count / n_mc,
            "SU_Outage": su_outage_count / n_mc,
            "Feasible_Count": 0,
            "Infeasible_Count": infeasible_count,
        }

    df = pd.DataFrame(feasible_records)

    return {
        "SNR_dB": snr_db,

        # averages over FEASIBLE solutions only
        "Avg_Pp": df["Pp"].mean(),
        "Avg_Psc": df["Psc"].mean(),
        "Avg_Psp": df["Psp"].mean(),
        "Avg_Rp": df["Rp"].mean(),
        "Avg_Rsc": df["Rsc"].mean(),
        "Avg_Rsp": df["Rsp"].mean(),
        "Avg_Rs": df["Rs"].mean(),
        "Avg_Reff_s": df["Reff_s"].mean(),
        "Avg_Layers_s": df["layers_s"].mean(),
        "Avg_PSNR_p": df["PSNR_p"].mean(),
        "Avg_PSNR_s": df["PSNR_s"].mean(),
        "Avg_QoE_p": df["QoE_p"].mean(),
        "Avg_QoE_s": df["QoE_s"].mean(),
        "Avg_QoE_sys": df["QoE_sys"].mean(),

        # outage over ALL realizations
        "PU_Outage": pu_outage_count / n_mc,
        "SU_Outage": su_outage_count / n_mc,

        "Feasible_Count": feasible_count,
        "Infeasible_Count": infeasible_count,
    }


def run_full_simulation() -> pd.DataFrame:
    """
    Run the full simulation over all SNR points.

    Output:
        pandas DataFrame with one row per SNR point
    """
    rows = []

    print("=== RUNNING FULL SIMULATION ===")
    print(f"Solver : {'SCA' if USE_SCA_SOLVER else 'GRID'}")
    print(f"SNR points : {list(SNR_dB_list)}")
    print(f"Monte Carlo per SNR : {N_MC}")

    for snr_db in SNR_dB_list:
        print(f"\n[INFO] Running SNR = {snr_db} dB ...")
        row = aggregate_snr_point(float(snr_db), n_mc=N_MC)
        rows.append(row)

        print(
            f"  Feasible = {row['Feasible_Count']}/{N_MC}, "
            f"Infeasible = {row['Infeasible_Count']}/{N_MC}, "
            f"Avg_QoE_sys = {row['Avg_QoE_sys']:.4f}, "
            f"PU_Outage = {row['PU_Outage']:.4f}, "
            f"SU_Outage = {row['SU_Outage']:.4f}"
        )

    df_results = pd.DataFrame(rows)
    return df_results


def save_results(df: pd.DataFrame, filename: str = "simulation_results.csv") -> Path:
    """
    Save results DataFrame to CSV.

    Output:
        saved file path
    """
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    df = run_full_simulation()
    print("\n=== FINAL RESULTS TABLE ===")
    print(df)

    save_path = save_results(df)
    print(f"\nSaved results to: {save_path}")