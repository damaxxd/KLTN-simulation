"""
outage_model.py
---------------

Khối 6: Outage definition

Input:
    - Rp
    - PSNR_s
    - thresholds from params

Output:
    - PU_outage
    - SU_outage

Notes:
    - PU outage follows a target-rate style threshold.
    - SU outage is defined via a quality threshold (PSNR),
      which is appropriate for the video-centric setting.
"""

from __future__ import annotations

from params import Rp_min, PSNR_s_min


def pu_outage(Rp: float, rate_threshold: float = Rp_min) -> int:
    """
    PU outage indicator.

    Rule:
        outage = 1 if Rp < Rp_min
        outage = 0 otherwise

    Input:
        Rp             : PU rate
        rate_threshold : PU minimum allowed rate

    Output:
        integer 0 or 1
    """
    return 1 if float(Rp) < float(rate_threshold) else 0


def su_outage(PSNR_s: float, psnr_threshold: float = PSNR_s_min) -> int:
    """
    SU outage indicator.

    Rule:
        outage = 1 if PSNR_s < PSNR_s_min
        outage = 0 otherwise

    Input:
        PSNR_s         : SU reconstructed quality
        psnr_threshold : minimum acceptable PSNR

    Output:
        integer 0 or 1
    """
    return 1 if float(PSNR_s) < float(psnr_threshold) else 0


def summarize_outage(Rp: float, PSNR_s: float) -> dict[str, int]:
    """
    Convenience wrapper.

    Output keys:
        PU_outage
        SU_outage
    """
    return {
        "PU_outage": pu_outage(Rp),
        "SU_outage": su_outage(PSNR_s),
    }


if __name__ == "__main__":
    Rp_test_1 = 50000.0
    Rp_test_2 = 20000.0
    PSNR_s_1 = 31.5
    PSNR_s_2 = 27.0

    print("=== OUTAGE MODEL TEST ===")
    print("Case 1:", summarize_outage(Rp_test_1, PSNR_s_1))
    print("Case 2:", summarize_outage(Rp_test_2, PSNR_s_2))