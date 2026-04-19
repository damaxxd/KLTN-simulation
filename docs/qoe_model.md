# QoE Model

## PSNR Model

PU:

PSNR_p is computed from the fitted rate-to-PSNR model.

SU:

PSNR_s is computed from the effective SVC rate `Reff_s` using the same fitted
inverse rate-to-PSNR model as PU:

PSNR_s = -10 log10(theta_s / (Reff_s - beta_s) - alpha_s) + 20 log10(255)

The SVC layer model still determines `Reff_s` and the decoded layer count, but
QoE no longer maps PSNR_s directly from the layer count.

## QoE Definition

PU:

QoE_p = log10(1 + PSNR_p)

SU:

QoE_s = log10(1 + PSNR_s)

System:

QoE_sys = 0.5 * QoE_p + 0.5 * QoE_s
