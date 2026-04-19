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
    Average quality/rate/power metrics are computed over feasible Monte Carlo
    realizations only. PU/SU outage statistics are computed over all channel
    realizations using the optimized solution outage event. Failed optimizer
    outcomes are counted as outage events.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import numpy as np

if __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from params import (
    SNR_dB_list, N_MC, USE_SCA_SOLVER,
    Psc_nonopt, Pp_nonopt, Psp_nonopt,
    Pp_max,
    w_p, w_s,
)
from channel import sigma2_from_snr_db, sample_pu_su_gains
from power_solver_grid import solve_power_grid
from access_baselines import compute_baseline_rates, solve_baseline_power_grid
from paper_reference_baselines import REFERENCE_SCHEMES, solve_reference_power_grid
from rate_model import compute_all_rates
from svc_abstraction import summarize_svc_state
from quality_model import summarize_quality
from outage_core import pu_outage, solution_outage


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

KQ2_BASELINE_SCHEMES = ["OMA", "NOMA", "CR_NOMA"]

KQ2_COMPARISON_COLUMNS = [
    "SNR_dB",
    "Avg_QoE_sys_Proposed",
    "Avg_QoE_sys_OMA",
    "Avg_QoE_sys_NOMA",
    "Avg_QoE_sys_CR_NOMA",
    "PU_Outage_Proposed",
    "PU_Outage_OMA",
    "PU_Outage_NOMA",
    "PU_Outage_CR_NOMA",
    "SU_Outage_Proposed",
    "SU_Outage_OMA",
    "SU_Outage_NOMA",
    "SU_Outage_CR_NOMA",
    "Feasible_Count_Proposed",
    "Feasible_Count_OMA",
    "Feasible_Count_NOMA",
    "Feasible_Count_CR_NOMA",
]

KQ5_REFERENCE_COLUMNS = [
    "SNR_dB",
    "Avg_PSNR_sys_Proposed",
    "Avg_PSNR_sys_Feng_RSMA_UAV",
    "Avg_PSNR_sys_He_CR_NOMA_MTCC",
    "Avg_PSNR_p_Proposed",
    "Avg_PSNR_p_Feng_RSMA_UAV",
    "Avg_PSNR_p_He_CR_NOMA_MTCC",
    "Avg_PSNR_s_Proposed",
    "Avg_PSNR_s_Feng_RSMA_UAV",
    "Avg_PSNR_s_He_CR_NOMA_MTCC",
    "Avg_Layers_s_Proposed",
    "Avg_Layers_s_Feng_RSMA_UAV",
    "Avg_Layers_s_He_CR_NOMA_MTCC",
    "Feasible_Count_Proposed",
    "Feasible_Count_Feng_RSMA_UAV",
    "Feasible_Count_He_CR_NOMA_MTCC",
]

KQ6_OUTAGE_COLUMNS = [
    "SNR_dB",
    "PU_Outage",
    "SU_Outage",
    "Feasible_Count",
    "Infeasible_Count",
]


def weighted_psnr(PSNR_p: float, PSNR_s: float) -> float:
    """Weighted average PSNR used in KQ1 comparison."""
    return float(w_p * PSNR_p + w_s * PSNR_s)


def _mean_or_nan(records: list[dict], key: str) -> float:
    """Return the mean value for a key, or NaN when no record is available."""
    if not records:
        return np.nan
    return float(pd.DataFrame(records)[key].mean())


def _ratio_or_nan(numerator: float, denominator: int) -> float:
    """Return a ratio, or NaN when the denominator has no feasible samples."""
    if denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


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
    outage = solution_outage(
        Rp=phy["Rp"],
        Rsc=phy["Rsc"],
        Reff_s=svc["Reff_s"],
        layers_s=svc["layers_s"],
    )

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

    baselines = {
        scheme: solve_baseline_power_grid(
            scheme=scheme,
            gp=gp,
            gs=gs,
            sigma2=sigma2,
        )
        for scheme in KQ2_BASELINE_SCHEMES
    }
    kq5_references = {
        scheme: solve_reference_power_grid(
            scheme=scheme,
            gp=gp,
            gs=gs,
            sigma2=sigma2,
        )
        for scheme in REFERENCE_SCHEMES
    }

    return {
        "opt": sol,
        "nonopt": nonopt,
        "baselines": baselines,
        "kq5_references": kq5_references,
        "gp": gp,
        "gs": gs,
        "sigma2": sigma2,
    }


def aggregate_snr_point(snr_db: float, n_mc: int = N_MC) -> dict:
    """
    Aggregate all Monte Carlo realizations for one SNR point.

    VERY IMPORTANT:
    - Average metrics (power, rates, PSNR, QoE, layers) are computed
      over feasible solutions only.
    - PU/SU outage statistics are computed over all channel realizations
      from the optimized solution. If no feasible allocation exists, PU outage
      is evaluated on the PU-only max-power link, while SU is counted as
      outage because no BL-feasible allocation was found.
    """
    feasible_records = []
    nonopt_records = []
    baseline_records = {scheme: [] for scheme in KQ2_BASELINE_SCHEMES}
    kq5_reference_records = {scheme: [] for scheme in REFERENCE_SCHEMES}

    pu_outage_count = 0
    su_outage_count = 0
    infeasible_count = 0
    baseline_pu_outage_count = {scheme: 0 for scheme in KQ2_BASELINE_SCHEMES}
    baseline_su_outage_count = {scheme: 0 for scheme in KQ2_BASELINE_SCHEMES}

    for _ in range(n_mc):
        realization = run_single_realization(snr_db)
        sol = realization["opt"]
        nonopt_records.append(realization["nonopt"])

        for scheme, reference_sol in realization["kq5_references"].items():
            if reference_sol is not None:
                kq5_reference_records[scheme].append(reference_sol)

        for scheme, baseline_sol in realization["baselines"].items():
            if baseline_sol is None:
                baseline_su_outage_count[scheme] += 1
                max_power_rates = compute_baseline_rates(
                    scheme=scheme,
                    gp=realization["gp"],
                    gs=realization["gs"],
                    Pp=Pp_max,
                    Ps=0.0,
                    sigma2=realization["sigma2"],
                )
                baseline_pu_outage_count[scheme] += pu_outage(max_power_rates["Rp"])
                continue

            baseline_records[scheme].append(baseline_sol)
            baseline_pu_outage_count[scheme] += int(baseline_sol["PU_outage"])
            baseline_su_outage_count[scheme] += int(baseline_sol["SU_outage"])

        if sol is None:
            infeasible_count += 1
            pu_only = compute_all_rates(
                gp=realization["gp"],
                gs=realization["gs"],
                Pp=Pp_max,
                Psc=0.0,
                Psp=0.0,
                sigma2=realization["sigma2"],
            )
            pu_outage_count += pu_outage(pu_only["Rp"])
            su_outage_count += 1
            continue

        opt_outage = solution_outage(
            Rp=sol["Rp"],
            Rsc=sol["Rsc"],
            Reff_s=sol["Reff_s"],
            layers_s=sol["layers_s"],
        )
        pu_outage_count += int(opt_outage["PU_outage"])
        su_outage_count += int(opt_outage["SU_outage"])
        sol = {
            **sol,
            "PU_outage": int(opt_outage["PU_outage"]),
            "SU_outage": int(opt_outage["SU_outage"]),
            "bl_decodable": bool(opt_outage["bl_decodable"]),
        }
        feasible_records.append(sol)

    feasible_count = len(feasible_records)
    df_nonopt = pd.DataFrame(nonopt_records)

    baseline_qoe_avg = {}
    baseline_feasible_count = {}
    for scheme in KQ2_BASELINE_SCHEMES:
        baseline_feasible_count[scheme] = len(baseline_records[scheme])
        baseline_qoe_avg[scheme] = _mean_or_nan(baseline_records[scheme], "QoE_sys")
    proposed_pu_outage = pu_outage_count / n_mc
    proposed_su_outage = su_outage_count / n_mc
    baseline_pu_outage = {
        scheme: float(baseline_pu_outage_count[scheme]) / float(n_mc)
        for scheme in KQ2_BASELINE_SCHEMES
    }
    baseline_su_outage = {
        scheme: float(baseline_su_outage_count[scheme]) / float(n_mc)
        for scheme in KQ2_BASELINE_SCHEMES
    }

    nonopt_psnr_p = df_nonopt["PSNR_p"].mean()
    nonopt_psnr_s = df_nonopt["PSNR_s"].mean()
    nonopt_psnr_avg = weighted_psnr(nonopt_psnr_p, nonopt_psnr_s)
    kq5_ref_psnr_p = {
        scheme: _mean_or_nan(kq5_reference_records[scheme], "PSNR_p")
        for scheme in REFERENCE_SCHEMES
    }
    kq5_ref_psnr_s = {
        scheme: _mean_or_nan(kq5_reference_records[scheme], "PSNR_s")
        for scheme in REFERENCE_SCHEMES
    }
    kq5_ref_psnr_sys = {
        scheme: weighted_psnr(kq5_ref_psnr_p[scheme], kq5_ref_psnr_s[scheme])
        for scheme in REFERENCE_SCHEMES
    }
    kq5_ref_layers = {
        scheme: _mean_or_nan(kq5_reference_records[scheme], "layers_s")
        for scheme in REFERENCE_SCHEMES
    }
    kq5_ref_feasible_count = {
        scheme: len(kq5_reference_records[scheme])
        for scheme in REFERENCE_SCHEMES
    }

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
            "PU_Outage": proposed_pu_outage,
            "SU_Outage": proposed_su_outage,
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

            "Avg_QoE_sys_Proposed": np.nan,
            "Avg_QoE_sys_OMA": baseline_qoe_avg["OMA"],
            "Avg_QoE_sys_NOMA": baseline_qoe_avg["NOMA"],
            "Avg_QoE_sys_CR_NOMA": baseline_qoe_avg["CR_NOMA"],
            "PU_Outage_Proposed": proposed_pu_outage,
            "PU_Outage_OMA": baseline_pu_outage["OMA"],
            "PU_Outage_NOMA": baseline_pu_outage["NOMA"],
            "PU_Outage_CR_NOMA": baseline_pu_outage["CR_NOMA"],
            "SU_Outage_Proposed": proposed_su_outage,
            "SU_Outage_OMA": baseline_su_outage["OMA"],
            "SU_Outage_NOMA": baseline_su_outage["NOMA"],
            "SU_Outage_CR_NOMA": baseline_su_outage["CR_NOMA"],
            "Feasible_Count_Proposed": 0,
            "Feasible_Count_OMA": baseline_feasible_count["OMA"],
            "Feasible_Count_NOMA": baseline_feasible_count["NOMA"],
            "Feasible_Count_CR_NOMA": baseline_feasible_count["CR_NOMA"],

            "Avg_PSNR_sys_Proposed": np.nan,
            "Avg_PSNR_sys_Feng_RSMA_UAV": kq5_ref_psnr_sys["Feng_RSMA_UAV"],
            "Avg_PSNR_sys_He_CR_NOMA_MTCC": kq5_ref_psnr_sys["He_CR_NOMA_MTCC"],
            "Avg_PSNR_p_Proposed": np.nan,
            "Avg_PSNR_p_Feng_RSMA_UAV": kq5_ref_psnr_p["Feng_RSMA_UAV"],
            "Avg_PSNR_p_He_CR_NOMA_MTCC": kq5_ref_psnr_p["He_CR_NOMA_MTCC"],
            "Avg_PSNR_s_Proposed": np.nan,
            "Avg_PSNR_s_Feng_RSMA_UAV": kq5_ref_psnr_s["Feng_RSMA_UAV"],
            "Avg_PSNR_s_He_CR_NOMA_MTCC": kq5_ref_psnr_s["He_CR_NOMA_MTCC"],
            "Avg_Layers_s_Proposed": np.nan,
            "Avg_Layers_s_Feng_RSMA_UAV": kq5_ref_layers["Feng_RSMA_UAV"],
            "Avg_Layers_s_He_CR_NOMA_MTCC": kq5_ref_layers["He_CR_NOMA_MTCC"],
            "Feasible_Count_Feng_RSMA_UAV": kq5_ref_feasible_count["Feng_RSMA_UAV"],
            "Feasible_Count_He_CR_NOMA_MTCC": kq5_ref_feasible_count["He_CR_NOMA_MTCC"],
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

        # outage over all channel realizations
        "PU_Outage": proposed_pu_outage,
        "SU_Outage": proposed_su_outage,

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

        "Avg_QoE_sys_Proposed": df["QoE_sys"].mean(),
        "Avg_QoE_sys_OMA": baseline_qoe_avg["OMA"],
        "Avg_QoE_sys_NOMA": baseline_qoe_avg["NOMA"],
        "Avg_QoE_sys_CR_NOMA": baseline_qoe_avg["CR_NOMA"],
        "PU_Outage_Proposed": proposed_pu_outage,
        "PU_Outage_OMA": baseline_pu_outage["OMA"],
        "PU_Outage_NOMA": baseline_pu_outage["NOMA"],
        "PU_Outage_CR_NOMA": baseline_pu_outage["CR_NOMA"],
        "SU_Outage_Proposed": proposed_su_outage,
        "SU_Outage_OMA": baseline_su_outage["OMA"],
        "SU_Outage_NOMA": baseline_su_outage["NOMA"],
        "SU_Outage_CR_NOMA": baseline_su_outage["CR_NOMA"],
        "Feasible_Count_Proposed": feasible_count,
        "Feasible_Count_OMA": baseline_feasible_count["OMA"],
        "Feasible_Count_NOMA": baseline_feasible_count["NOMA"],
        "Feasible_Count_CR_NOMA": baseline_feasible_count["CR_NOMA"],

        "Avg_PSNR_sys_Proposed": opt_psnr_avg,
        "Avg_PSNR_sys_Feng_RSMA_UAV": kq5_ref_psnr_sys["Feng_RSMA_UAV"],
        "Avg_PSNR_sys_He_CR_NOMA_MTCC": kq5_ref_psnr_sys["He_CR_NOMA_MTCC"],
        "Avg_PSNR_p_Proposed": opt_psnr_p,
        "Avg_PSNR_p_Feng_RSMA_UAV": kq5_ref_psnr_p["Feng_RSMA_UAV"],
        "Avg_PSNR_p_He_CR_NOMA_MTCC": kq5_ref_psnr_p["He_CR_NOMA_MTCC"],
        "Avg_PSNR_s_Proposed": opt_psnr_s,
        "Avg_PSNR_s_Feng_RSMA_UAV": kq5_ref_psnr_s["Feng_RSMA_UAV"],
        "Avg_PSNR_s_He_CR_NOMA_MTCC": kq5_ref_psnr_s["He_CR_NOMA_MTCC"],
        "Avg_Layers_s_Proposed": df["layers_s"].mean(),
        "Avg_Layers_s_Feng_RSMA_UAV": kq5_ref_layers["Feng_RSMA_UAV"],
        "Avg_Layers_s_He_CR_NOMA_MTCC": kq5_ref_layers["He_CR_NOMA_MTCC"],
        "Feasible_Count_Feng_RSMA_UAV": kq5_ref_feasible_count["Feng_RSMA_UAV"],
        "Feasible_Count_He_CR_NOMA_MTCC": kq5_ref_feasible_count["He_CR_NOMA_MTCC"],
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


def save_kq2_comparison(df: pd.DataFrame, filename: str = "kq2_qoe_scheme_comparison.csv") -> Path:
    """
    Save the KQ2 QoE-vs-SNR scheme comparison table.
    """
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / filename
    df.loc[:, KQ2_COMPARISON_COLUMNS].to_csv(out_path, index=False)
    return out_path


def save_kq5_reference_comparison(
    df: pd.DataFrame,
    filename: str = "kq5_psnr_reference_comparison.csv",
) -> Path:
    """
    Save the KQ5 PSNR-equivalent paper-reference comparison table.
    """
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / filename
    df.loc[:, KQ5_REFERENCE_COLUMNS].to_csv(out_path, index=False)
    return out_path


def save_kq6_outage(df: pd.DataFrame, filename: str = "kq6_outage_probability.csv") -> Path:
    """
    Save the KQ6 PU/SU outage probability table.
    """
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / filename
    df.loc[:, KQ6_OUTAGE_COLUMNS].to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    df = run_full_simulation()
    print("\n=== FINAL RESULTS TABLE ===")
    print(df.loc[:, MAIN_RESULT_COLUMNS])

    print("\n=== KQ1 POWER / PSNR COMPARISON ===")
    print(df.loc[:, KQ1_COMPARISON_COLUMNS])

    print("\n=== KQ2 QOE SCHEME COMPARISON ===")
    print(df.loc[:, KQ2_COMPARISON_COLUMNS])

    print("\n=== KQ5 PSNR REFERENCE COMPARISON ===")
    print(df.loc[:, KQ5_REFERENCE_COLUMNS])

    print("\n=== KQ6 OUTAGE PROBABILITY ===")
    print(df.loc[:, KQ6_OUTAGE_COLUMNS])

    save_path = save_results(df)
    print(f"\nSaved results to: {save_path}")

    kq1_save_path = save_kq1_comparison(df)
    print(f"Saved KQ1 comparison to: {kq1_save_path}")

    kq2_save_path = save_kq2_comparison(df)
    print(f"Saved KQ2 comparison to: {kq2_save_path}")

    kq5_save_path = save_kq5_reference_comparison(df)
    print(f"Saved KQ5 reference comparison to: {kq5_save_path}")

    kq6_save_path = save_kq6_outage(df)
    print(f"Saved KQ6 outage probability to: {kq6_save_path}")
