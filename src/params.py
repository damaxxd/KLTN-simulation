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
# 3) SVC BITRATES
# =========================================================
# User-measured QCIF cumulative bitrate operating points:
#   QP40 -> 33.3576 kbps
#   QP34 -> 66.1392 kbps
#   QP28 -> 129.7792 kbps
#   QP22 -> 258.5376 kbps
#
# Converted into SVC layer increments:
#   BL  = 33.3576
#   EL1 = 66.1392 - 33.3576 = 32.7816
#   EL2 = 129.7792 - 66.1392 = 63.6400
#   EL3 = 258.5376 - 129.7792 = 128.7584
#
# Unit in code: bps
svc_layers: np.ndarray = np.array(
    [33357.6, 32781.6, 63640.0, 128758.4],
    dtype=float,
)
svc_cum: np.ndarray = np.cumsum(svc_layers)

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
Pp_max: float = 2.0     # W
Ps_max: float = 2.0     # W, total SU budget: Psc + Psp <= Ps_max

# Non-optimized baseline power vector in the requested order:
# P = (Psc, Pp, Psp)
Psc_nonopt: float = 1.5
Pp_nonopt: float = 2.0
Psp_nonopt: float = 0.5

# =========================================================
# 6) SIMULATION GRID
# =========================================================
SNR_dB_list: np.ndarray = np.arange(0, 31, 2)
N_MC: int = 300

# =========================================================
# 7) OUTAGE / QUALITY THRESHOLDS
# =========================================================
# PU protection threshold:
# thesis-specific assumption:
# PU should at least achieve a base-layer-comparable rate
Rp_min: float = float(r_BL)

# SU outage threshold in terms of reconstructed quality
PSNR_s_min: float = 34.0  # dB

# =========================================================
# 8) OBJECTIVE WEIGHTS
# =========================================================
# Since PU must be protected, we prioritize PU more strongly.
w_p: float = 0.7
w_s: float = 0.3

# =========================================================
# 9) RATE-PSNR FITTING PARAMETERS
# =========================================================
# These parameters are needed because the paper provides the
# model form but not fixed numeric coefficients.
# Therefore, they are simulation fitting parameters.
theta_p: float = 2.0e7
alpha_p: float = 1.0
beta_p: float = 0.0

theta_s: float = 2.2e7
alpha_s: float = 1.0
beta_s: float = 0.0

# =========================================================
# 10) SOLVER SETTINGS
# =========================================================
USE_SCA_SOLVER: bool = False
GRID_POINTS_PU: int = 10
GRID_POINTS_SU: int = 11

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
# QoE_s = PSNR_s
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
    print(f"SVC layers (bps)       : {svc_layers}")
    print(f"SVC cumulative (bps)   : {svc_cum}")
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
