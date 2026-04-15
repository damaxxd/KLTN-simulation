# Optimization Problem

Maximize:
QoE_sys

Subject to:
- Rp ≥ Rp_min
- Rsc ≥ r_BL
- Pp ≤ Pp_max
- Psc + Psp ≤ Ps_max

Solver:
- SCA (Successive Convex Approximation)
- Linearize log terms
- Solve via SLSQP