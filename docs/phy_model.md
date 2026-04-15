# PHY Model

## Channel
- Rayleigh fading:
  gp ~ Exp(1)
  gs ~ Exp(1)

## SINR

PU:
SINR_p = gp * Pp / (gs*(Psc + Psp) + σ²)

SU common:
SINR_sc = gs * Psc / (gs*Psp + gp*Pp + σ²)

SU private:
SINR_sp = gs * Psp / (gp*Pp + σ²)

## Rate (AMC Model)
R = c1 * B * log2(1 + SINR / c2)