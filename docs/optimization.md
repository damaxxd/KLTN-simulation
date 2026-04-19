# Optimization Problem

Maximize:

QoE_sys = 0.5 * log10(1 + PSNR_p) + 0.5 * log10(1 + PSNR_s)

Subject to:

- Rp >= Rp_min
- Rsc >= r_BL
- Pp <= Pp_max
- Psc + Psp <= Ps_max
- gs * Psp <= tau for Proposed CR-RSMA after common-stream SIC

Solver:

- SCA linearizes log terms and solves each surrogate with SLSQP.
- Grid warm-start with local refinement is used as the robust comparison solver.
