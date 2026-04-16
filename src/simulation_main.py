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

from params import (
    SNR_dB_list, N_MC, USE_SCA_SOLVER,
    Psc_nonopt, Pp_nonopt, Psp_nonopt,
    w_p, w_s,
)
from channel import sigma2_from_snr_db, sample_pu_su_gains
from power_solver_grid import solve_power_grid
from rate_model import compute_all_rates
from svc_abstraction import summarize_svc_state
from quality_model import summarize_quality
from outage_model import summarize_outage


MAIN_RESULT_COLUMNS = [
    "SNR_dB",
    "Avg_Pp",
    "Avg_Psc",
    "Avg_Psp",
    "Avg_Rp",
    "Avg_Rsc",
    "Avg_Rsp",
    "Avg_Rs",
    "Avg_Reff_s",
    "Avg_Layers_s",
    "Avg_PSNR_p",
    "Avg_PSNR_s",
    "Avg_QoE_p",
    "Avg_QoE_s",
    "Avg_QoE_sys",
    "PU_Outage",
    "SU_Outage",
    "Feasible_Count",
    "Infeasible_Count",
]

KQ1_COMPARISON_COLUMNS = [
    "SNR_dB",
    "Psc_opt",
    "Pp_opt",
    "Psp_opt",
    "Psc_nonopt",
    "Pp_nonopt",
    "Psp_nonopt",
    "PSNR_p_opt",
    "PSNR_s_opt",
    "PSNR_avg_opt",
    "PSNR_p_nonopt",
    "PSNR_s_nonopt",
    "PSNR_avg_nonopt",
    "Delta_PSNR_avg",
]


def weighted_psnr(PSNR_p: float, PSNR_s: float) -> float:
    """Weighted average PSNR used in KQ1 comparison."""
    return float(w_p * PSNR_p + w_s * PSNR_s)


def evaluate_power_vector(
    gp: float,
    gs: float,
    sigma2: float,
    Pp: float,
    Psc: float,
    Psp: float,
) -> dict:
    """
    Evaluate a fixed power vector with the same PHY/SVC/QoE/outage pipeline
    used by the optimizer.
    """
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
    outage = summarize_outage(phy["Rp"], quality["PSNR_s"])

    return {
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
    }


def run_single_realization(snr_db: float, d_pu: float = 1.2, d_su: float = 1.0) -> dict:
    """
    Run one Monte Carlo realization at one SNR point.

    Input:
        snr_db : current SNR in dB
        d_pu   : PU distance
        d_su   : SU distance

    Output:
        dictionary with optimized and non-optimized results
    """
    sigma2 = sigma2_from_snr_db(snr_db)
    gp, gs = sample_pu_su_gains(d_pu=d_pu, d_su=d_su)

    if USE_SCA_SOLVER:
        from power_solver_sca import solve_power_sca
        sol = solve_power_sca(gp=gp, gs=gs, sigma2=sigma2)
    else:
        sol = solve_power_grid(gp=gp, gs=gs, sigma2=sigma2)

    nonopt = evaluate_power_vector(
        gp=gp,
        gs=gs,
        sigma2=sigma2,
        Pp=Pp_nonopt,
        Psc=Psc_nonopt,
        Psp=Psp_nonopt,
    )

    return {
        "opt": sol,
        "nonopt": nonopt,
    }


def aggregate_snr_point(snr_db: float, n_mc: int = N_MC) -> dict:
    """
    Aggregate all Monte Carlo realizations for one SNR point.

    VERY IMPORTANT:
    - Average metrics (power, rates, PSNR, QoE, layers) are computed
      over feasible solutions only.
    - Outage statistics are computed over ALL realizations.
    """
    feasible_records = []
    nonopt_records = []

    # outage counters over ALL realizations
    pu_outage_count = 0
    su_outage_count = 0
    infeasible_count = 0

    for _ in range(n_mc):
        realization = run_single_realization(snr_db)
        sol = realization["opt"]
        nonopt_records.append(realization["nonopt"])

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
    df_nonopt = pd.DataFrame(nonopt_records)

    nonopt_psnr_p = df_nonopt["PSNR_p"].mean()
    nonopt_psnr_s = df_nonopt["PSNR_s"].mean()
    nonopt_psnr_avg = weighted_psnr(nonopt_psnr_p, nonopt_psnr_s)

    if feasible_count == 0:
        opt_psnr_avg = np.nan
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

            "Psc_opt": np.nan,
            "Pp_opt": np.nan,
            "Psp_opt": np.nan,
            "Psc_nonopt": Psc_nonopt,
            "Pp_nonopt": Pp_nonopt,
            "Psp_nonopt": Psp_nonopt,
            "PSNR_p_opt": np.nan,
            "PSNR_s_opt": np.nan,
            "PSNR_avg_opt": opt_psnr_avg,
            "PSNR_p_nonopt": nonopt_psnr_p,
            "PSNR_s_nonopt": nonopt_psnr_s,
            "PSNR_avg_nonopt": nonopt_psnr_avg,
            "Delta_PSNR_avg": np.nan,
        }

    df = pd.DataFrame(feasible_records)
    opt_psnr_p = df["PSNR_p"].mean()
    opt_psnr_s = df["PSNR_s"].mean()
    opt_psnr_avg = weighted_psnr(opt_psnr_p, opt_psnr_s)

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

        "Psc_opt": df["Psc"].mean(),
        "Pp_opt": df["Pp"].mean(),
        "Psp_opt": df["Psp"].mean(),
        "Psc_nonopt": Psc_nonopt,
        "Pp_nonopt": Pp_nonopt,
        "Psp_nonopt": Psp_nonopt,
        "PSNR_p_opt": opt_psnr_p,
        "PSNR_s_opt": opt_psnr_s,
        "PSNR_avg_opt": opt_psnr_avg,
        "PSNR_p_nonopt": nonopt_psnr_p,
        "PSNR_s_nonopt": nonopt_psnr_s,
        "PSNR_avg_nonopt": nonopt_psnr_avg,
        "Delta_PSNR_avg": opt_psnr_avg - nonopt_psnr_avg,
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
    columns = MAIN_RESULT_COLUMNS if filename == "simulation_results.csv" else df.columns
    df.loc[:, columns].to_csv(out_path, index=False)
    return out_path


def save_kq1_comparison(df: pd.DataFrame, filename: str = "kq1_power_psnr_comparison.csv") -> Path:
    """
    Save the KQ1 power/PSNR comparison table without feasibility count columns.
    """
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / filename
    df.loc[:, KQ1_COMPARISON_COLUMNS].to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    df = run_full_simulation()
    print("\n=== FINAL RESULTS TABLE ===")
    print(df.loc[:, MAIN_RESULT_COLUMNS])

    print("\n=== KQ1 POWER / PSNR COMPARISON ===")
    print(df.loc[:, KQ1_COMPARISON_COLUMNS])

    save_path = save_results(df)
    print(f"\nSaved results to: {save_path}")

    kq1_save_path = save_kq1_comparison(df)
    print(f"Saved KQ1 comparison to: {kq1_save_path}")
