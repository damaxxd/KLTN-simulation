# Traceability mo hinh va ket qua mo phong

Tai lieu nay giai thich y nghia cac bien/cau lenh chinh trong source code, cong thuc dang duoc dung, nguon tham khao cua cong thuc/gia tri, va cach chung tao ra cac bang/duong cong KQ1, KQ2, KQ3, KQ5, KQ6.

Luu y quan trong: khong phai moi dong code deu lay truc tiep tu journal. Code hien tai gom 4 nhom nguon:

- Paper/journal/source chuan: cong thuc AMC rate, rate-PSNR model, Rayleigh fading, SVC layering, OMA/NOMA/CR-NOMA/CR-RSMA principles.
- Du lieu do/fit cua project: SVC bitrate/PSNR, bo tham so alpha/beta/theta fit tu video.
- Mo hinh de xuat cua de tai: mapping CR-RSMA common stream -> SVC base layer, private stream -> enhancement layers; objective QoE-aware.
- Gia dinh mo phong/AI-built abstraction: grid search solver, baseline video-efficiency penalty, so lan Monte Carlo, power grid, mot so nguong outage.

## 1. Khoi tham so `params.py`

| Bien trong code | Gia tri hien tai | Y nghia | Nguon/giai trinh |
|---|---:|---|---|
| `RNG_SEED` | `42` | Seed de ket qua Monte Carlo lap lai duoc | Gia dinh mo phong |
| `resolution`, `width`, `height` | `QCIF`, `176`, `144` | Dinh dang video | SVC/video test setting |
| `fps` | `30` | Frame rate | Theo setting video mo phong |
| `qp_points` | `[40,34,28,22]` | Cac diem QP dung de tao RD/SVC operating points | Theo qua trinh encode/fit video |
| `svc_cum` | `[71377.6, 142985.6, 287816.0, 544243.2]` bps | Bitrate tich luy de decode BL, BL+EL1, BL+EL1+EL2, all layers | Du lieu do/fit cua project, khong phai hang so journal |
| `svc_layers` | `diff([0]+svc_cum)` | Incremental layer rates: BL, EL1, EL2, EL3 | SVC abstraction, tinh tu `svc_cum` |
| `svc_y_psnr` | `[29.8718,33.1740,36.9828,41.1876]` dB | PSNR tai cac operating points | Du lieu do/fit cua project |
| `B` | `140e3` Hz | Bandwidth dung trong cong thuc rate | Tham khao quality-driven SVC/NOMA paper va proposal |
| `c1` | `0.905` | He so dieu chinh AMC | Tham khao quality-driven SVC/NOMA paper/proposal |
| `c2` | `1.34` | SNR gap cua AMC | Tham khao quality-driven SVC/NOMA paper/proposal |
| `eta` | `2.0` | Path-loss exponent | Thong so channel thong dung/simulation assumption |
| `Pp_max`, `Ps_max` | `1.0`, `1.0` W | Gioi han cong suat PU/SU | Gia dinh mo phong hien tai |
| `Psc_nonopt`, `Pp_nonopt`, `Psp_nonopt` | `0.75`, `1.0`, `0.25` | Vector cong suat khong toi uu de so sanh KQ1 | Baseline thesis-specific |
| `SNR_dB_list` | `0:2:30` dB | Tap SNR tren truc hoanh | Gia dinh mo phong |
| `N_MC` | `300` | So mau Monte Carlo moi diem SNR | Gia dinh mo phong |
| `Rp_min` | `r_BL = 71377.6` bps | Nguong bao ve PU | Main-comparison setting: PU QoS target aligned with the SVC base-layer bitrate |
| `w_p`, `w_s` | `0.5`, `0.5` | Trong so QoE he thong | Gia dinh objective cua de tai |
| `theta_p, alpha_p, beta_p` | Fit params PU | Tham so rate-PSNR cho PU | Dang fit tu du lieu video, paper chi cho dang ham |
| `theta_s, alpha_s, beta_s` | Fit params SU | Tham so rate-PSNR cho SU | Dang fit tu du lieu video, paper chi cho dang ham |

Can noi voi thay: paper cho dang cong thuc PSNR-rate, nhung alpha/beta/theta la fitting parameters cua video/dataset. Chung khong phai hang so co dinh lay truc tiep tu journal.

## 2. Khoi kenh `channel.py`

### Noise tu SNR

Code:

```python
snr_lin = 10.0 ** (snr_db / 10.0)
sigma2 = P_ref / snr_lin
```

Cong thuc:

\[
SNR_{lin}=10^{SNR_{dB}/10},\quad \sigma^2=\frac{P_{ref}}{SNR_{lin}}
\]

Y nghia:

- Truc hoanh cua hau het cac hinh la `SNR_dB`.
- Khi SNR tang, `sigma2` giam, nen rate tang, PSNR/QoE tang, outage giam.

Nguon:

- Cong thuc doi dB/linear va noise normalization la cong thuc co ban trong wireless communication textbook.

### Rayleigh fading + path loss

Code:

```python
g ~ CN(0,1)
h = g / sqrt(1 + d^eta)
gain = |h|^2
```

Y nghia:

- `gp`: channel power gain cua PU.
- `gs`: channel power gain cua SU.
- Moi Monte Carlo draw sinh mot cap `gp, gs`.

Nguon:

- Rayleigh fading va path loss la mo hinh kenh co ban trong wireless communication.
- Dang `1 + d^eta` la simulation abstraction de tranh singularity khi distance nho.

## 3. Khoi PHY/rate `rate_model.py`

### Rate theo AMC

Code:

```python
R = c1 * B * log2(1 + SINR / c2)
```

Cong thuc:

\[
R = c_1 B \log_2\left(1+\frac{\gamma}{c_2}\right)
\]

Y nghia:

- `B`: bandwidth.
- `c1`: AMC rate adjustment.
- `c2`: SNR gap.
- `gamma`: SINR.
- Output la bit/s.

Nguon:

- Tham khao quality-driven SVC/NOMA paper/proposal.
- Neu khong co AMC thi cong thuc Shannon co dang \(B\log_2(1+\gamma)\). O day dung ban AMC-adjusted.

### SINR cua proposed CR-RSMA uplink

Code:

```python
SINR_p = gp * Pp / (gs * Psp + sigma2)
SINR_sc = gs * Psc / (gs * Psp + gp * Pp + sigma2)
SINR_sp = gs * Psp / sigma2
```

Giai thich:

- `Pp`: cong suat PU.
- `Psc`: cong suat SU common stream.
- `Psp`: cong suat SU private stream.
- `gp`: gain PU -> BS/UAV receiver.
- `gs`: gain SU -> BS/UAV receiver.

Thu tu giai ma dang mo phong:

1. Decode SU common stream truoc, nen common stream thay nhieu nhieu: PU + SU private + noise.
2. Sau khi common stream duoc cancel, decode PU; PU con bi residual interference tu SU private.
3. Sau khi PU duoc cancel, decode SU private; private stream chi con noise.

Nguon/giai trinh:

- Nguyen ly RSMA/SIC lay tu RSMA uplink/CR-RSMA literature.
- Dang SINR cu the la mo hinh de xuat/abstraction cua project, khong copy nguyen xi tu mot paper duy nhat.

### Tong rate

Code:

```python
Rp = R(SINR_p)
Rsc = R(SINR_sc)
Rsp = R(SINR_sp)
Rs = Rsc + Rsp
```

Y nghia:

- `Rp`: throughput/rate cua PU.
- `Rsc`: SU common rate, gan voi base layer.
- `Rsp`: SU private rate, gan voi enhancement layers.
- `Rs`: tong SU PHY rate.

## 4. Khoi SVC `svc_abstraction.py`

### Mapping CR-RSMA -> SVC

Code:

```python
Reff_s = min(Rsc, r_BL) + min(Rsp, sum(EL))
```

Cong thuc:

\[
R_{eff,s} = \min(R_{sc}, r_{BL}) + \min(R_{sp}, r_{EL1}+r_{EL2}+r_{EL3})
\]

Y nghia:

- Base layer BL phai duoc bao ve bang common stream.
- Enhancement layers EL duoc truyen bang private stream.
- Neu `Rsc < r_BL`, SU khong decode duoc base layer day du.

Nguon:

- SVC layering lay tu SVC standard/literature.
- Mapping common -> BL va private -> EL la phan de xuat cua de tai.

### So layer decode duoc

Code:

```python
layers_s = sum(Reff_s >= svc_cum)
```

Y nghia:

- `layers_s=1`: decode BL.
- `layers_s=2`: BL+EL1.
- `layers_s=3`: BL+EL1+EL2.
- `layers_s=4`: tat ca layers.

Dung cho:

- Auxiliary curve: average decoded layers vs SNR.
- QoE/outage SU.

## 5. Khoi PSNR/QoE `quality_model.py`

### PSNR tu MSE

Code:

```python
PSNR = 10 * log10(255^2 / MSE)
```

Nguon:

- Cong thuc PSNR chuan trong image/video processing textbooks va SVC literature.

### Rate-PSNR model

Code dang inverse:

```python
R = theta / (alpha + 255^2 * 10^(-Q/10)) + beta
```

va:

```python
PSNR = -10*log10(theta/(R-beta) - alpha) + 20*log10(255)
```

Cong thuc:

\[
R(Q)=\frac{\theta}{\alpha + 255^2 10^{-Q/10}}+\beta
\]

\[
Q(R)=-10\log_{10}\left(\frac{\theta}{R-\beta}-\alpha\right)+20\log_{10}(255)
\]

Y nghia:

- `Q` la PSNR.
- `R` la bitrate/rate.
- `theta`, `alpha`, `beta` la fitted parameters.
- Code dung `PSNR_MIN_DB = 0` de tranh PSNR am khi rate nam ngoai mien fit.

Nguon:

- Dang ham rate-PSNR tham khao paper Enabling Quality-Driven SVC/NOMA.
- Gia tri `theta/alpha/beta` hien tai la fitting parameters cua project, khong phai gia tri co dinh tu paper.

### QoE

Code hien tai:

```python
QoE_p = PSNR_p
QoE_s = PSNR_s
QoE_sys = w_p * QoE_p + w_s * QoE_s
```

Cong thuc:

\[
QoE_{sys}=w_p QoE_p+w_s QoE_s
\]

voi:

\[
QoE_p=PSNR_p,\quad QoE_s=PSNR_s
\]

Y nghia:

- QoE duoc bieu dien bang PSNR de de giai thich va de lien he truc tiep voi human perceptual quality.
- Neu bat `USE_SVC_AWARE_QOE=True`, code co the them layer reward va rate reward, nhung hien tai dang tat.

Nguon:

- Weighted utility la objective de xuat cua de tai.
- QoE=PSNR la thesis-specific simplification dua tren viec PSNR la metric chat luong video pho bien.

## 6. Khoi outage

Code hien tai da rut gon outage thanh cac file can thiet:

- `outage_core.py`: outage chinh cua project, dung cho KQ6.
- `interference_protection.py`: nguong SINR, tau, va rang buoc bao ve PU trong solver.

Hai file outage phu truoc do (`outage_model.py`, `paper_outage_reference.py`) da duoc xoa vi khong can cho muc tieu KQ6.

### PU outage

Code:

```python
PU_outage = 1 if Rp < Rp_min else 0
```

Cong thuc:

\[
P_{out,p} = Pr(R_p < R_{p,min})
\]

Y nghia:

- PU duoc bao ve theo threshold rate.
- Neu PU khong dat `Rp_min`, mau Monte Carlo do tinh la outage.

Nguon:

- Target-rate outage la cong thuc thong dung trong wireless/CR/NOMA papers.

### SU outage

Code main:

```python
SU_outage = 1 if BL not decodable else 0
```

Dieu kien BL decodable:

```python
Rsc >= r_BL and Reff_s >= r_BL and layers_s >= 1
```

Y nghia:

- SU chi duoc xem la phuc vu thanh cong neu base layer decode duoc.
- Enhancement layer giup tang QoE/PSNR nhung khong phai dieu kien toi thieu de tranh outage.

Nguon:

- SVC literature: base layer la layer bat buoc, enhancement layer phu thuoc BL.
- Dieu kien outage nay la de xuat cua thesis, ap dung SVC vao CR-RSMA.

### Interference threshold tau

Code:

```python
gamma_target = c2 * (2^(Rp_min/(c1*B)) - 1)
tau = gp * Pp / gamma_target - sigma2
```

Y nghia:

- `gamma_target`: SINR toi thieu de PU dat `Rp_min`.
- `tau`: muc nhieu toi da SU duoc gay ra ma PU van dat QoS.

Nguon:

- Tu viec dao nguoc cong thuc AMC rate.
- Logic interference-temperature/QoS protection tham khao CR-NOMA/CR-RSMA papers.

## 7. Khoi toi uu cong suat `power_solver_grid.py`

### Bien toi uu

\[
P=(P_{s,c}, P_p, P_{s,p})
\]

Trong code luu theo thu tu:

```python
Pp, Psc, Psp
```

nhung KQ1 xuat theo yeu cau:

```text
(Psc, Pp, Psp)
```

### Rang buoc

1. Gioi han tong cong suat SU:

\[
P_{s,c}+P_{s,p}\le P_{s,max}
\]

Code:

```python
if Psc + Psp > Ps_max: reject
```

2. SU khong im lang:

\[
P_{s,c}+P_{s,p}>0
\]

3. Bao ve PU bang residual interference:

\[
g_s P_{s,p}\le \tau
\]

Code:

```python
su_respects_pu_residual_interference_budget(...)
```

4. PU rate constraint:

\[
R_p \ge R_{p,min}
\]

5. BL decodability:

\[
R_{sc}\ge r_{BL}
\]

### Objective

Code:

```python
objective = QoE_sys
```

Cong thuc:

\[
\max_{P_p,P_{s,c},P_{s,p}} QoE_{sys}
\]

voi:

\[
QoE_{sys}=0.5PSNR_p+0.5PSNR_s
\]

Nguon:

- Objective cross-layer QoE-aware la phan de xuat cua de tai.
- Grid search la phuong phap so/implementation, khong phai cong thuc journal.

## 8. Baseline KQ2 `access_baselines.py`

### OMA

Code:

```python
Rp = 0.5 * R(gp*Pp/sigma2)
Rs = 0.5 * R(gs*Ps/sigma2)
```

Y nghia:

- PU va SU chia tai nguyen 50/50.
- Khong co nhieu cheo.

Nguon:

- OMA principle tu communication textbooks/survey papers.
- He so 0.5 la gia dinh equal-time/equal-resource.

### NOMA/CR-NOMA

Code:

```python
SINR_p = gp*Pp / (gs*Ps + sigma2)
SINR_s = gs*Ps / (gp*Pp + sigma2)
```

Y nghia:

- Hai user cung dung tai nguyen, co nhieu lan nhau.
- CR-NOMA them constraint `Rp >= Rp_min`.

Nguon:

- NOMA/CR-NOMA principle tu NOMA/CR-NOMA literature.
- Dang baseline la abstraction de so sanh, khong reproduce chinh xac mot paper.

### Video efficiency penalty

Code:

```python
OMA: 0.78
NOMA: 0.74
CR_NOMA: 0.82
Reff_s = efficiency * Rs
```

Y nghia:

- Baseline khong co co che common stream bao ve BL nhu proposed.
- Raw PHY rate khong chuyen 1:1 thanh useful SVC video rate.

Nguon:

- Gia dinh mo phong/thesis-specific abstraction.
- Duoc dung de the hien uu the cross-layer SVC-aware cua proposed.
- Can noi ro voi thay: day khong phai hang so journal; la penalty mo phong de baseline khong duoc huong SVC-aware protection.

## 9. KQ1: bang cong suat toi uu va PSNR opt/non-opt

File:

```text
results/kq1_power_psnr_comparison.csv
```

Truc/bang:

- Cot `SNR_dB`: dieu kien kenh.
- Cot `Psc_opt`, `Pp_opt`, `Psp_opt`: trung binh vector cong suat toi uu.
- Cot `PSNR_avg_opt`: PSNR trung binh khi dung vector toi uu.
- Cot `PSNR_avg_nonopt`: PSNR trung binh khi dung vector co dinh khong toi uu.
- Cot `Delta_PSNR_avg`: loi ich cua toi uu cong suat.

Cong thuc PSNR trung binh:

\[
PSNR_{avg}=w_pPSNR_p+w_sPSNR_s
\]

Y nghia duong/bang:

- SNR tang -> noise giam -> rate tang -> PSNR tang.
- Neu `Delta_PSNR_avg > 0`, power allocation de xuat co loi hon non-opt.

Nguon:

- Power allocation objective la de xuat cua de tai.
- PSNR-rate model tham khao quality-driven SVC/NOMA paper.

## 10. KQ2: QoE trung binh vs SNR proposed/OMA/NOMA/CR-NOMA

File:

```text
results/kq2_qoe_scheme_comparison.csv
```

Truc hoanh:

\[
SNR\ (dB)
\]

Truc tung:

\[
QoE_{sys}=w_pQoE_p+w_sQoE_s
\]

voi hien tai:

\[
QoE_p=PSNR_p,\quad QoE_s=PSNR_s
\]

Y nghia:

- Duong proposed: CR-RSMA + SVC-aware power allocation.
- Duong OMA/NOMA/CR-NOMA: baseline access principles.
- Neu proposed cao hon, nghia la cross-layer power allocation + SVC mapping mang lai QoE video cao hon.

Nguon:

- OMA/NOMA/CR-NOMA principles tu literature.
- Video-efficiency penalty cua baseline la mo hinh mo phong thesis-specific.
- Proposed objective la cong thuc de xuat.

Can noi voi thay:

- KQ2 khong phai reproduce exact OMA/NOMA papers.
- No la comparison theo nguyen ly truy nhap chung, quy doi qua cung PSNR/QoE pipeline.

## 11. KQ3: QoE PU va SU vs SNR trong proposed

File nguon:

```text
results/simulation_results.csv
```

Truc hoanh:

\[
SNR\ (dB)
\]

Truc tung:

\[
QoE_p=PSNR_p,\quad QoE_s=PSNR_s
\]

Y nghia:

- PU QoE cao/on dinh hon do PU duoc uu tien bao ve.
- SU QoE tang theo SNR vi co them rate de decode EL.
- Khoang cach PU-SU the hien trade-off trong power allocation.

Nguon:

- PU protection la Cognitive Radio principle.
- SU layered improvement la SVC principle.
- Objective la de xuat cua de tai.

## 12. KQ5: PSNR trung binh vs SNR proposed va throughput-max references

File:

```text
results/kq5_psnr_reference_comparison.csv
```

Truc hoanh:

\[
SNR\ (dB)
\]

Truc tung:

\[
PSNR_{sys}=w_pPSNR_p+w_sPSNR_s
\]

Y nghia:

- Proposed toi uu truc tiep QoE/PSNR nen PSNR cao.
- Reference throughput-max methods toi uu rate/throughput, sau do moi quy doi sang PSNR.
- Neu throughput cao nhung khong phuc vu dung SVC BL/EL thi PSNR-equivalent co the thap.

Nguon:

- Reference methods la adapted baselines tu RSMA-throughput-max va CR-NOMA-throughput-max papers.
- Phan quy doi throughput sang PSNR dung rate-PSNR model cua quality-driven SVC/NOMA paper.

Can noi voi thay:

- Day la PSNR-equivalent comparison, khong phai exact reproduction of original papers.
- Original papers co the khong bao cao PSNR; minh dung cung channel/power/SVC setting cua thesis de quy doi cong bang sang PSNR.

## 13. KQ6: outage probability PU/SU vs SNR

File:

```text
results/kq6_outage_probability.csv
```

Truc hoanh:

\[
SNR\ (dB)
\]

Truc tung:

\[
P_{out}=Pr(outage)
\]

PU outage:

\[
P_{out,p}=Pr(R_p<R_{p,min})
\]

SU outage:

\[
P_{out,s}=Pr(\text{BL not decodable})
\]

BL not decodable khi:

\[
R_{sc}<r_{BL}
\]

hoac:

\[
R_{eff,s}<r_{BL}
\]

hoac:

\[
layers_s<1
\]

Y nghia:

- SNR tang -> noise giam -> outage giam.
- PU outage thap hon SU outage vi PU duoc bao ve truoc.
- SU outage cao o SNR thap vi SU phai thoa ca PU protection va BL decoding.

Nguon:

- Target-rate outage la mo hinh co ban trong wireless/CR/NOMA papers.
- BL-decoding outage la de xuat ap dung SVC cho thesis.

## 14. Cau tra loi ngan gon khi thay hoi "cong thuc nao lay tu dau?"

- Kenh Rayleigh + path loss: lay tu mo hinh wireless communication co ban.
- Rate AMC \(R=c_1B\log_2(1+\gamma/c_2)\): tham khao quality-driven SVC/NOMA paper/proposal.
- PSNR \(10\log_{10}(255^2/MSE)\): cong thuc video/image processing chuan.
- Rate-PSNR \(R=\theta/(\alpha+255^2 10^{-Q/10})+\beta\): tham khao Enabling Quality-Driven SVC/NOMA paper; tham so fit tu video/project.
- SVC BL/EL va cumulative layers: tham khao SVC standard/literature; bitrate/PSNR la du lieu do/fit cua project.
- CR-RSMA common/private stream: dua tren RSMA/CR-RSMA principle; mapping common->BL, private->EL la phan de xuat cua thesis.
- PU QoS constraint \(R_p\ge R_{p,min}\): Cognitive Radio principle.
- Outage \(Pr(R<R_{min})\): outage analysis chuan trong wireless.
- Baseline OMA/NOMA/CR-NOMA: lay nguyen ly chung tu literature, khong reproduce exact paper.
- Baseline video-efficiency coefficients: gia dinh mo phong de phan anh baseline khong SVC-aware; can ghi ro la thesis-specific simulation setting.

## 15. Diem can than trong bao cao

1. Khong noi "tat ca cong thuc deu lay nguyen xi tu paper". Dung hon la: mot so cong thuc lay tu literature, mot so la adaptation/de xuat.
2. Cac he so `BASELINE_VIDEO_EFFICIENCY` khong phai tham so journal. Neu thay hoi, phai giai thich la penalty mo phong de baseline khong duoc huong SVC-aware protection.
3. Grid search solver la implementation de tim nghiem, khong phai dong gop ly thuyet chinh. Neu muon nghiem chuan hon, co the thay bang SCA.
4. KQ5 la comparison quy doi sang PSNR. Neu paper goc khong bao cao PSNR, phai ghi "PSNR-equivalent adapted comparison".
5. `N_MC=300` co the lam outage curve hoi rung. Neu can hinh dep hon, tang `N_MC` len 1000 hoac 2000.
