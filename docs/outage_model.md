# Outage Model

The KQ6 outage curves use a Monte Carlo version of the QoS-protected
rate-splitting principle from the uplink CR-NOMA paper.

## PU protection

The PU target rate is converted to an AMC-adjusted SINR target:

gamma_p_target = c2 * (2^(Rp_min / (c1 * B)) - 1)

For each channel realization, the maximum received SU interference that can be
tolerated by the PU is:

tau = max(0, gp * Pp / gamma_p_target - sigma2)

The PU outage indicator follows the paper's OMA-protected interpretation:

PU_outage = 1 if gp * Pp < gamma_p_target * sigma2, otherwise 0.

Therefore, SU transmission can be limited or rejected without counting a PU
outage when the PU itself is still supportable in the no-SU condition.

## SU outage

The SU outage indicator follows the target-rate event used by the CR-NOMA
rate-splitting paper. For the thesis SVC model, the SU target rate is the
base-layer rate:

R_s,target = r_BL

For the proposed CR-RSMA scheme, the SU achievable rate is computed with the
paper's three RS cases:

- Case I: tau > 0 and Ps * gs <= tau
- Case II: tau > 0 and Ps * gs > tau
- Case III: tau = 0

The SU outage event is:

SU_outage = 1 if R_s,RS < r_BL, otherwise 0.

PSNR_s_min is no longer used for outage probability. Video quality and QoE are
still evaluated separately through the measured 30 fps SVC model.
