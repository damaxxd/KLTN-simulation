"""
params.py
---------

Khối 1: Parameter block

Input:
    None

Output:
    Export all simulation constants used by the project:
    - video parameters
    - SVC bitrates
    - PHY parameters
    - power budgets
    - SNR grid
    - outage thresholds

Notes on sources:
    - B, c1, c2, eta, fps, gop_size, qp_points:
        adapted from the quality-driven NOMA video paper
    - SVC layered structure:
        based on SVC literature
    - svc_layers:
        derived from the user's measured QCIF bitrates
    - Rp_min, PSNR_s_min, N_MC:
        simulation assumptions / thesis-specific settings
"""

from __future__ import annotations

import numpy as np

# =========================================================
# 1) RANDOMNESS / REPRODUCIBILITY
# =========================================================
RNG_SEED: int = 42

# =========================================================
# 2) VIDEO PARAMETERS
# =========================================================
resolution: str = "QCIF"
width: int = 176
height: int = 144
fps: int = 30
gop_size: int = 8
qp_points: list[int] = [40, 34, 28, 22]

# =========================================================
# 3) MEASURED SVC OPERATING POINTS
# =========================================================
# User-measured cumulative bitrate / Y-PSNR points used by this simulation:
#   QP40 ->  71.3776 kbps, Y-PSNR 29.8718 dB
#   QP34 -> 142.9856 kbps, Y-PSNR 33.1740 dB
#   QP28 -> 287.8160 kbps, Y-PSNR 36.9828 dB
#   QP22 -> 544.2432 kbps, Y-PSNR 41.1876 dB
#
# Converted into SVC layer increments:
#   BL  = 71.3776 kbps
#   EL1 = 142.9856 - 71.3776 = 71.6080 kbps
#   EL2 = 287.8160 - 142.9856 = 144.8304 kbps
#   EL3 = 544.2432 - 287.8160 = 256.4272 kbps
#
# Unit in code: bitrate in bps, PSNR in dB.
svc_qp_points: list[int] = [40, 34, 28, 22]
svc_cum: np.ndarray = np.array(
    [71377.6, 142985.6, 287816.0, 544243.2],
    dtype=float,
)
svc_layers: np.ndarray = np.diff(
    np.concatenate(([0.0], svc_cum))
)
svc_y_psnr: np.ndarray = np.array(
    [29.8718, 33.1740, 36.9828, 41.1876],
    dtype=float,
)

# Convenience aliases
r_BL: float = float(svc_layers[0])
r_EL1: float = float(svc_layers[1])
r_EL2: float = float(svc_layers[2])
r_EL3: float = float(svc_layers[3])

# =========================================================
# 4) PHY PARAMETERS
# =========================================================
# From the quality-driven NOMA video paper
B: float = 140e3        # Hz
c1: float = 0.905       # AMC rate adjustment factor
c2: float = 1.34        # AMC SNR gap
eta: float = 2.0        # path-loss exponent

# =========================================================
# 5) POWER BUDGET
# =========================================================
# Per-user power budgets for the current comparison setting.
Pp_max: float = 1.0     # W
Ps_max: float = 1.0     # W, total SU budget: Psc + Psp <= Ps_max

# Non-optimized baseline power vector in the requested order:
# P = (Psc, Pp, Psp)
Psc_nonopt: float = 0.75
Pp_nonopt: float = 1.0
Psp_nonopt: float = 0.25

# =========================================================
# 6) SIMULATION GRID
# =========================================================
SNR_dB_list: np.ndarray = np.arange(0, 31, 2)
N_MC: int = 1000

# =========================================================
# 7) OUTAGE / QUALITY THRESHOLDS
# =========================================================
# PU protection threshold:
# main-comparison setting:
# PU should at least achieve a base-layer-comparable rate, so we align the PU
# QoS target with the SVC base-layer bitrate used by the SU.
Rp_min: float = float(r_BL)

# SU outage threshold in terms of reconstructed quality
PSNR_s_min: float = 34.0  # dB

# =========================================================
# 8) OBJECTIVE WEIGHTS
# =========================================================
# PU and SU contribute equally to the system QoE objective.
w_p: float = 0.5
w_s: float = 0.5

# =========================================================
# 9) RATE-PSNR FITTING PARAMETERS
# =========================================================
# These parameters are needed because the paper provides the
# model form but not fixed numeric coefficients.
# Therefore, they are simulation fitting parameters.
#pu: foreman
theta_p: float = 1282878.251174431
alpha_p: float = 0.895858948403827
beta_p: float = 12889.069811351561

#su: soccer
theta_s: float = 4793840.045889435
alpha_s: float = 3.98203901019467
beta_s: float = 5773.96320163053

# =========================================================
# 10) SOLVER SETTINGS
# =========================================================
USE_SCA_SOLVER: bool = True
GRID_POINTS_PU: int = 10
GRID_POINTS_SU: int = 11
GRID_REFINEMENT_ENABLED: bool = True
GRID_REFINE_POINTS_PU: int = 7
GRID_REFINE_POINTS_SU: int = 9

# SCA placeholders for later use
SCA_MAX_ITER: int = 30
SCA_TOL: float = 1e-4

# =========================================================
# 11) REFERENCE POWER FOR NOISE COMPUTATION
# =========================================================
P_ref: float = 1.0  # W

# =========================================================
# 12) SVC-AWARE QoE SETTINGS
# =========================================================
# Current comparison setting:
# QoE_p = PSNR_p and QoE_s = PSNR_s
USE_SVC_AWARE_QOE: bool = False

# Reward for activating more SVC layers
lambda_layer: float = 4.0

# Optional reward for effective video rate (keep 0 for now)
lambda_rate: float = 0.0

def print_parameter_summary() -> None:
    """Pretty-print the main simulation settings."""
    print("=== PARAMETER SUMMARY ===")
    print(f"Resolution             : {resolution} ({width}x{height})")
    print(f"FPS / GOP              : {fps} / {gop_size}")
    print(f"QP points              : {qp_points}")
    print(f"SVC QP points          : {svc_qp_points}")
    print(f"SVC layers (bps)       : {svc_layers}")
    print(f"SVC cumulative (bps)   : {svc_cum}")
    print(f"SVC Y-PSNR (dB)        : {svc_y_psnr}")
    print(f"B (Hz)                 : {B}")
    print(f"AMC (c1, c2)           : ({c1}, {c2})")
    print(f"Path-loss exponent     : {eta}")
    print(f"Power budgets (W)      : Pp_max={Pp_max}, Ps_max={Ps_max}")
    print(
        "Non-opt baseline (W)   : "
        f"Psc={Psc_nonopt}, Pp={Pp_nonopt}, Psp={Psp_nonopt}"
    )
    print(f"SNR grid (dB)          : {SNR_dB_list}")
    print(f"Monte Carlo runs       : {N_MC}")
    print(f"PU min rate (bps)      : {Rp_min}")
    print(f"SU PSNR outage (dB)    : {PSNR_s_min}")
    print(f"Objective weights      : w_p={w_p}, w_s={w_s}")
    print(f"Use SCA solver         : {USE_SCA_SOLVER}")


if __name__ == "__main__":
    print_parameter_summary()
