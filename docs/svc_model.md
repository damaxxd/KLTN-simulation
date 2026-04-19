# SVC Model

Layers:
- BL (base layer)
- EL1, EL2, EL3

Mapping:
- Rsc → BL
- Rsp → EL

Effective rate:
Reff_s = min(Rsc, r_BL) + min(Rsp, r_EL_total)

Layer decoding:
Compare Reff_s with cumulative bitrate thresholds

Measured 30 fps operating points:

| Layer count | SVC point | Cumulative bitrate (bps) | Y-PSNR (dB) |
| --- | --- | ---: | ---: |
| 1 | QP40 / BL | 33357.6 | 30.9794 |
| 2 | QP34 / BL+EL1 | 66139.2 | 34.8523 |
| 3 | QP28 / BL+EL1+EL2 | 129779.2 | 38.5330 |
| 4 | QP22 / BL+EL1+EL2+EL3 | 258537.6 | 42.0377 |
