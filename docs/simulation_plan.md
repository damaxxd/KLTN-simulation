# Simulation Plan

## Monte Carlo
- 300 runs per SNR

## SNR range
0 → 30 dB

## Outputs

KQ1: Power vs SNR  
KQ2: QoE_sys vs SNR  
KQ3: QoE_p & QoE_s  
KQ5: PSNR vs SNR  
KQ6: Outage vs SNR  

## Outage definition
PU:
QoS-protected outage follows the CR-NOMA rate-splitting paper. For each
realization, compute the AMC-adjusted PU SINR target and the interference
threshold:

tau = max(0, gp * Pp / gamma_p_target - sigma2)

PU outage is counted only when the PU cannot meet Rp_min in the no-SU
OMA-protected condition.

SU:
SU outage follows the CR-NOMA RS paper target-rate event. In this thesis, the
target rate is the SVC base-layer rate:

R_s,target = r_BL

For Proposed CR-RSMA, compute the paper-style RS achievable rate over the three
cases defined by tau:
- Case I: tau > 0 and Ps * gs <= tau
- Case II: tau > 0 and Ps * gs > tau
- Case III: tau = 0

SU outage is counted when:

R_s,RS < r_BL

PSNR_s_min is not used for outage probability.
