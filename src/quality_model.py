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
    - PU and SU PSNR use the inverse rate-PSNR relation used by the
      quality-driven NOMA video paper.
    - SU effective SVC rate is converted directly to PSNR.
    - QoE is configured in params.py. The thesis-default setting uses
      QoE = PSNR for direct interpretability.
"""

from __future__ import annotations

import numpy as np

from params import (
    w_p, w_s,
    svc_cum,
    theta_p, alpha_p, beta_p,
    theta_s, alpha_s, beta_s,
    USE_SVC_AWARE_QOE, lambda_layer, lambda_rate,
)

PSNR_MIN_DB: float = 0.0


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
    psnr = -10.0 * np.log10(inside) + 20.0 * np.log10(255.0)
    return float(max(psnr, PSNR_MIN_DB))


def psnr_pu_from_rate(Rp: float) -> float:
    return psnr_from_rate_paper(Rp, theta_p, alpha_p, beta_p)


def psnr_su_from_layers(layers_s: int) -> float:
    """
    Deprecated compatibility helper.

    SU PSNR is now computed from effective SVC rate, not directly from the
    decoded layer count. Call psnr_su_from_rate(Reff_s) when possible.
    """
    layers = int(layers_s)
    if layers <= 0:
        return psnr_su_from_rate(0.0)

    idx = min(layers, len(svc_cum)) - 1
    return psnr_su_from_rate(float(svc_cum[idx]))


def psnr_su_from_rate(Reff_s: float) -> float:
    """
    Return SU PSNR from effective SVC rate using the fitted rate-PSNR model.
    """
    return psnr_from_rate_paper(Reff_s, theta_s, alpha_s, beta_s)


def qoe_from_psnr(PSNR: float) -> float:
    """Return QoE from PSNR.

    The main thesis comparison uses QoE = PSNR. Keeping the conversion in one
    helper makes it explicit and prevents mixed PSNR/log-PSNR objectives.
    """
    return float(max(float(PSNR), 0.0))


def qoe_pu_from_psnr(PSNR_p: float) -> float:
    """
    QoE_p = PSNR_p
    """
    return qoe_from_psnr(PSNR_p)


def qoe_su_from_psnr_layers(
    PSNR_s: float,
    layers_s: int,
    Reff_s: float,
    max_layers: int = 4,
) -> float:
    """
    QoE_s = PSNR_s by default.

    Optional SVC-aware terms can be enabled from params.py when the thesis
    needs an explicit reward for decoded layers or effective video rate.
    """
    qoe = qoe_from_psnr(PSNR_s)
    if USE_SVC_AWARE_QOE:
        layer_reward = lambda_layer * (float(layers_s) / float(max_layers))
        rate_reward = lambda_rate * float(Reff_s) / max(float(svc_cum[-1]), 1e-12)
        qoe += layer_reward + rate_reward
    return float(qoe)


def qoe_system(QoE_p: float, QoE_s: float) -> float:
    return w_p * QoE_p + w_s * QoE_s


def summarize_quality(Rp: float, Reff_s: float, layers_s: int) -> dict[str, float]:
    """
    Convenience wrapper to summarize all quality outputs.
    """
    _ = layers_s
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
        (0.0, 0),
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
