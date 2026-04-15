# QoE Model

## PSNR model
PSNR = a + b * log10(rate)

## QoE definition

PU:
QoE_p = PSNR_p

SU:
QoE_s = PSNR_s 
       + λ_layer * (layers_s / 4)
       + λ_rate * (Reff_s / Rmax)

System:
QoE_sys = w_p * QoE_p + w_s * QoE_s