"""
svc_abstraction.py
------------------

Khối 4: SVC abstraction

Input:
    - Rsc, Rsp from the PHY rate model
    - svc_layers, svc_cum from params

Output:
    - Reff_s       : effective SVC video rate for SU
    - layers_s     : number of decodable layers
    - status_s     : binary vector of layer decoding status

Notes:
    - BL/EL layering is based on SVC literature.
    - The mapping common -> BL and private -> EL
      is thesis-specific modeling.
"""

from __future__ import annotations

import numpy as np

from params import svc_layers, svc_cum


def rsma_effective_video_rate(Rsc: float, Rsp: float, layers: np.ndarray = svc_layers) -> float:
    """
    Compute SU effective video rate under the RSMA-to-SVC mapping.

    Model:
        Reff_s = min(Rsc, r_BL) + min(Rsp, r_EL_total)

    Input:
        Rsc   : common-stream rate
        Rsp   : private-stream rate
        layers: SVC layer-increment vector [BL, EL1, EL2, EL3]

    Output:
        effective video rate Reff_s
    """
    r_BL = float(layers[0])
    r_EL_total = float(np.sum(layers[1:]))

    rate_bl = min(float(Rsc), r_BL)
    rate_el = min(float(Rsp), r_EL_total)

    return rate_bl + rate_el


def decodable_layers(rate_eff: float, cumulative_rates: np.ndarray = svc_cum) -> int:
    """
    Count how many cumulative SVC operating points are decodable.

    Rule:
        layers = count(rate_eff >= cumulative_rate[k])

    Input:
        rate_eff         : effective video rate
        cumulative_rates : cumulative SVC thresholds

    Output:
        integer number of decodable layers
    """
    return int(np.sum(rate_eff >= cumulative_rates))


def layer_status(rate_eff: float, cumulative_rates: np.ndarray = svc_cum) -> np.ndarray:
    """
    Return binary per-layer decoding status.

    Input:
        rate_eff         : effective video rate
        cumulative_rates : cumulative SVC thresholds

    Output:
        array([0/1, 0/1, 0/1, 0/1])
    """
    return (rate_eff >= cumulative_rates).astype(int)


def summarize_svc_state(Rsc: float, Rsp: float) -> dict[str, object]:
    """
    Convenience wrapper to summarize SVC-side outputs.

    Output keys:
        Reff_s
        layers_s
        status_s
    """
    Reff_s = rsma_effective_video_rate(Rsc, Rsp, svc_layers)
    layers_s = decodable_layers(Reff_s, svc_cum)
    status_s = layer_status(Reff_s, svc_cum)

    return {
        "Reff_s": Reff_s,
        "layers_s": layers_s,
        "status_s": status_s,
    }


if __name__ == "__main__":
    Rsc_test = 40000.0
    Rsp_test = 80000.0

    out = summarize_svc_state(Rsc_test, Rsp_test)

    print("=== SVC ABSTRACTION TEST ===")
    print(f"svc_layers : {svc_layers}")
    print(f"svc_cum    : {svc_cum}")
    for k, v in out.items():
        print(f"{k:8s}: {v}")