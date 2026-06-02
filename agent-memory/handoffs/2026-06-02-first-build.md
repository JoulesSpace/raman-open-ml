---
title: First build - scaffold, baselines, trust layer, domain-shift study
type: handoff
date: 2026-06-02
tags: [handoff]
---

# Handoff 2026-06-02: first build

## State (what exists and works)

Repo at `C:\Users\julia\Development\raman-open-ml`, git `main`. A public-data-only
Raman ML toolkit (no proprietary data used).

**Data** (git-ignored, fetched by `scripts/download_data.py`):
- bacteria-ID: 60k reference + 3k finetune + 3k test, 30 isolates, 1000 wn.
- polystyrene LoD: 48 spectra, 8 sizes x 6 concentrations, with per-point SD.

**Package** `src/raman_ml/`: preprocessing (ALS/arPLS/airPLS, SNV/MSC, SG deriv,
norms, resample), datasets, models (sklearn + 1D-CNN + 1D-ResNet, augmentation,
MC-dropout, C-Mixup), augment (SpecAugment + Raman), uncertainty (conformal
intervals/sets, temperature, ECE), ood (MSP/energy/Mahalanobis + AUROC/FPR95),
calibration_transfer (PDS). No third-party deps beyond numpy/scipy/sklearn/torch.

**Scripts / benchmarks** (outputs in `benchmarks/`):
- `run_classification.py`: LogReg/LinearSVM/RF/1D-CNN flat comparison.
- `run_quantification.py`: PLSR/PCR/SVR/RF/1D-CNN, relative-conc target.
- `run_domain_shift.py`: in-dist / cross-domain / adapted + temperature + RAPS.
- `run_openset.py`: hold out isolates, reject with MSP/energy/Mahalanobis.
- `run_calibration_transfer.py`: simulated secondary instrument + PDS + jackknife+.

## Verified results

- Classification in-distribution: **1D-ResNet 0.941**, LogReg 0.919. Cross-domain
  0.557 / 0.480 (the shift). Adapted 0.759 / 0.806.
- **SOTA protocol (pretrain->finetune->test): heterogeneous 8-member ensemble
  (SE-ResNet + multi-scale) + TTA = 0.862**, beating Ho 2019 (0.822) and SANet
  2026 (0.861); below open-world SOTA 0.878. Plain SE-ResNet ens 0.852,
  multi-scale-only 0.844 - diversity + TTA cleared SANet. Next lever: SSL.
- Trust layer: ECE 0.131 -> 0.045 (temperature); conformal coverage 0.923
  (target 0.90), set size 16/30 under shift.
- Quantification (CV, relative log10 conc): RF 0.848, SVR 0.830, CNN 0.779,
  PLSR 0.682, PCR 0.589.
- Open-set (24 known + 6 unknown): closed-set acc 0.807; OOD AUROC ~0.73-0.75.
- Calibration transfer: PDS recovers SVR (-0.03 -> 0.48) and RF (0.39 -> 0.61).

## Key finding

bacteria-ID has a severe reference->test campaign shift ([[bacteria-id-domain-shift]]).
This reframed the project around domain-shift-aware eval + a trust layer
(decisions [[0004-trustworthy-ml-positioning]], [[0005-domain-shift-aware-evaluation]]).

## Next steps (not yet done)

- SSL masked-autoencoder pretraining (SMAE) for the low-data regime.
- CARS/VIP variable selection for PLS; GAN/diffusion augmentation for the tiny
  quantification set.
- Interpretability: integrated-gradients peak attribution.
- Deep ensembles wired as a first-class model (MC-dropout exists; ensemble loop
  not yet scripted).
- Package for PyPI; add a test suite; honest-limitations note covering the
  small-n conformal undercoverage on quantification (jackknife+ ~0.75 vs 0.90).
- Open-set AUROC is only ~0.75; try Objectosphere / feature-regularisation
  training (Lebron et al. 2024) to push it up.
