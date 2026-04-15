"""
quality_model.py
----------------

Khối 5: Video quality / QoE model

Input:
    - Rp from the PHY model (for PU)
    - Reff_s from the SVC abstraction (for SU)
    - layers_s from the SVC abstraction

Output:
    - PSNR_p
    - PSNR_s
    - QoE_p
    - QoE_s
    - QoE_sys

Model source:
    - PSNR definition and inverse rate-PSNR relation follow
      the quality-driven NOMA video paper.
    - Layer-aware QoE is thesis-specific, introduced to reflect
      SVC enhancement-layer benefits that plain PSNR does not fully capture.
"""

from __future__ import annotations

import numpy as np

from params import (
    w_p, w_s,
    USE_SVC_AWARE_QOE,
    lambda_layer, lambda_rate,
    svc_cum,
)

# =========================================================
# CALIBRATED FITTING PARAMETERS
# =========================================================
theta_p: float = 1.37e6
alpha_p: float = 0.0
beta_p: float = 0.0

theta_s: float = 1.37e6
alpha_s: float = 0.0
beta_s: float = 0.0


def psnr_from_mse(mse: float) -> float:
    mse = max(float(mse), 1e-12)
    return 10.0 * np.log10((255.0 ** 2) / mse)


def rate_from_psnr_paper(Q: float, theta: float, alpha: float, beta: float) -> float:
    term = alpha + (255.0 ** 2) * (10.0 ** (-Q / 10.0))
    term = max(term, 1e-12)
    return theta / term + beta


def psnr_from_rate_paper(R: float, theta: float, alpha: float, beta: float) -> float:
    R_safe = max(float(R), beta + 1e-9)
    inside = theta / (R_safe - beta) - alpha
    inside = max(inside, 1e-12)
    return -10.0 * np.log10(inside) + 20.0 * np.log10(255.0)


def psnr_pu_from_rate(Rp: float) -> float:
    return psnr_from_rate_paper(Rp, theta_p, alpha_p, beta_p)


def psnr_su_from_rate(Reff_s: float) -> float:
    return psnr_from_rate_paper(Reff_s, theta_s, alpha_s, beta_s)


def qoe_pu_from_psnr(PSNR_p: float) -> float:
    """
    For now:
        QoE_p = PSNR_p
    """
    return float(PSNR_p)


def qoe_su_from_psnr_layers(
    PSNR_s: float,
    layers_s: int,
    Reff_s: float,
    max_layers: int = 4,
) -> float:
    """
    Thesis-specific SVC-aware QoE:

        QoE_s = PSNR_s
                + lambda_layer * (layers_s / max_layers)
                + lambda_rate  * (Reff_s / R_max)

    For the current stage:
        lambda_rate = 0 by default
    """
    if not USE_SVC_AWARE_QOE:
        return float(PSNR_s)

    R_max = float(svc_cum[-1])
    layer_bonus = lambda_layer * (float(layers_s) / float(max_layers))
    rate_bonus = lambda_rate * (float(Reff_s) / max(R_max, 1e-9))

    return float(PSNR_s) + layer_bonus + rate_bonus


def qoe_system(QoE_p: float, QoE_s: float) -> float:
    return w_p * QoE_p + w_s * QoE_s


def summarize_quality(Rp: float, Reff_s: float, layers_s: int) -> dict[str, float]:
    """
    Convenience wrapper to summarize all quality outputs.
    """
    PSNR_p = psnr_pu_from_rate(Rp)
    PSNR_s = psnr_su_from_rate(Reff_s)

    QoE_p = qoe_pu_from_psnr(PSNR_p)
    QoE_s = qoe_su_from_psnr_layers(PSNR_s, layers_s, Reff_s)
    QoE_sys = qoe_system(QoE_p, QoE_s)

    return {
        "PSNR_p": PSNR_p,
        "PSNR_s": PSNR_s,
        "QoE_p": QoE_p,
        "QoE_s": QoE_s,
        "QoE_sys": QoE_sys,
    }


if __name__ == "__main__":
    test_points = [
        (33357.6, 1),
        (66139.2, 2),
        (129779.2, 3),
        (258537.6, 4),
    ]

    print("=== QUALITY MODEL TEST ===")
    for rate, layers in test_points:
        out = summarize_quality(Rp=rate, Reff_s=rate, layers_s=layers)
        print(
            f"Rate={rate:10.1f} bps, layers={layers} "
            f"-> PSNR_s={out['PSNR_s']:.4f}, QoE_s={out['QoE_s']:.4f}"
        )