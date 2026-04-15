# AI Agent Instructions (CR-RSMA + SVC Thesis)

## Project Summary
This project simulates a CR-RSMA uplink system with UAV-assisted communication and SVC video transmission.

## Core Objective
Maximize system QoE:
QoE_sys = w_p * QoE_p + w_s * QoE_s

Subject to:
- PU protection (Rp ≥ Rp_min)
- SVC base layer decodability
- Power constraints

---

## System Components
- 1 Primary User (PU)
- 1 Secondary User (SU)
- SU uses SVC (BL + EL1 + EL2 + EL3)
- RSMA uplink:
  - Common stream → BL
  - Private stream → EL

---

## Key Modeling Rules

### PHY layer
- Rayleigh fading channels
- SINR computed using interference model
- Rate uses AMC approximation:
  R = c1 * B * log2(1 + SINR / c2)

---

### SVC Mapping
- BL requires Rsc ≥ r_BL
- EL layers depend on Rsp
- Effective rate:
  Reff_s = min(Rsc, r_BL) + min(Rsp, sum(EL))

---

### QoE Model
- QoE_p = PSNR_p
- QoE_s = PSNR_s + λ_layer * (layers_s / 4) + λ_rate * normalized_rate

---

## Optimization
- Solver: SCA (Successive Convex Approximation)
- Variables:
  - Pp, Psc, Psp
- Constraints:
  - Psc + Psp ≤ Ps_max
  - Rp ≥ Rp_min
  - Rsc ≥ r_BL

---

## Coding Instructions
- Always use modular structure:
  - channel → rate → svc → quality → solver
- Do NOT hardcode parameters
- Always return full metrics:
  - power, rates, layers, QoE, outage

---

## Simulation Rules
- Monte Carlo simulation required
- Use full SNR sweep (0 → 30 dB)
- Output must include:
  - QoE vs SNR
  - Outage vs SNR
  - Power allocation vs SNR

---

## Priority Order
1. Ensure PU protection
2. Guarantee BL decoding
3. Maximize SU QoE via EL layers