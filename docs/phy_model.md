# PHY Model

## Channel

- Rayleigh fading:
  - gp ~ Exp(1)
  - gs ~ Exp(1)

## Proposed CR-RSMA SIC Order

The proposed receiver decodes streams in this order:

1. SU common stream.
2. PU stream after canceling the SU common stream.
3. SU private stream after canceling the PU stream.

## SINR

PU:

SINR_p = gp * Pp / (gs * Psp + sigma2)

SU common:

SINR_sc = gs * Psc / (gs * Psp + gp * Pp + sigma2)

SU private:

SINR_sp = gs * Psp / sigma2

## Rate (AMC Model)

R = c1 * B * log2(1 + SINR / c2)
