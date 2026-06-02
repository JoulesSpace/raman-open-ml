---
title: SOTA Raman ML, OSS landscape, and transferable methods (research synthesis)
type: reference
date: 2026-06-02
tags: [sota, literature, roadmap]
---

# Research synthesis (three parallel literature scans, 2026-06-02)

Cited primary sources backing the improvement roadmap. See [[bacteria-id-domain-shift]]
for the dataset finding that reframed the project.

## SOTA architectures (classification)

- **Ho et al. 2019, Nat. Commun. 10:4927** (bacteria-ID): 26-layer 1D-ResNet,
  **strided convs, NO pooling** (preserve peak positions), 5th-order poly
  baseline + 0-1 norm. 30-class **82.2%** (LR 75.7%, SVM 74.9%); antibiotic
  group 97.0%. https://www.nature.com/articles/s41467-019-12898-9
- **Benchmark 2026, arXiv:2601.16107**: unified protocol on bacteria-ID. SANet
  (multi-scale CNN) **86.1%** > Deep CNN 85.8% > RamanFormer 84.2% > RamanNet
  83.7% > vanilla Transformer 81.5%. Transformers underperform at this scale.
  Documents val->test shift (99%->74-80%).
- **Lebron et al. 2024, Chem. Biomed. Imaging** (open-world): ensemble of 5
  ResNets with **Squeeze-Excitation on the last block**, **87.8%** closed-set,
  plus open-set rejection. https://pubs.acs.org/doi/10.1021/cbmi.4c00007
- **RamanNet** (Ibtehaz 2023, arXiv:2201.09737): shift-invariant windowed-MLP.

## SSL / low-data

- **SMAE, arXiv:2504.16130 (2025)**: masked-autoencoder pretraining, 50% mask,
  patch 100. 83.9% on bacteria-ID with only 100 labels/class (+6.1 vs scratch);
  also denoises. Biggest wins in low-data regime.

## Augmentation

- **Frontiers Oncol. 2024 (PMC11219827)**: noise/shift help LINEAR models more
  than CNNs; mixup-of-raw weak alone; **GAN augmentation gives the biggest CNN
  gain**; best = combination. **C-Mixup (NeurIPS 2022, arXiv:2210.05775)**:
  label-distance-weighted mixup, the right mixup for regression/dilution series.
- Wu et al. 2021 Sci. Rep.: GAN aug +5.4 pts on 149 samples.

## Preprocessing

- **RamanSPy (Anal. Chem. 2024)** recommended pipeline: crop fingerprint
  700-1800 cm-1 -> cosmic-ray removal (Whitaker-Hayes) -> denoise -> ASLS
  baseline -> AUC norm. arPLS/airPLS (pybaselines) improve on plain ALS.
  SNV/MSC + Savitzky-Golay derivatives are the cheapest high-impact lever for
  small-data quantification (Rinnan 2009).
- CNNs can also learn baseline end-to-end (Liu 2017; Wahl 2020 single-step CNN
  preprocessing).

## The open problems (where SOTA still struggles) = our opportunity

1. **Open-set / OOD**: softmax is closed-world; ~100% FP on unknown species.
   Energy (Liu 2020), Mahalanobis (Lee 2018), Objectosphere/feature-reg
   (arXiv:2310.13723). Absent from every OSS Raman lib.
2. **Domain / instrument shift + calibration transfer**: the dominant real
   failure. PDS/DS (Bouveresse 1996), LoRA-CT (Anal. Chem. 2025,
   10.1021/acs.analchem.5c01846). Our bacteria-ID finding is a live example.
3. **Uncertainty**: conformal prediction (CQR Romano 2019; RAPS Angelopoulos
   2021) gives finite-sample coverage; deep ensembles (Lakshminarayanan 2017);
   temperature scaling + ECE. Only in isolated papers, no maintained lib.
4. **Reproducibility / fair benchmark**: numbers across papers not comparable.
5. **Interpretability**: peak attribution (saliency / integrated gradients).

## OSS landscape (the bar)

RamanSPy (BSD, leader: preprocessing breadth, datasets, pipelines) | rampy |
SpectroChemPy | chemometrics libs (PLS/VIP) | bacteria-ID/RamanNet (models) |
pybaselines (50+ baselines backend). **None ship integrated UQ, open-set/OOD,
or calibration transfer.** That trustworthy-ML + reproducibility layer is the
empty, defensible territory this repo targets.

## Roadmap (impact-to-effort)

P1 domain-shift-aware eval + adaptation (in-dist / cross-domain / adapted).
P2 1D-ResNet (SOTA arch) + on-the-fly augmentation + ensembles.
P3 open-set/OOD (Mahalanobis + energy, AUROC/FPR95 on held-out classes).
P4 conformal prediction (sets for clf, intervals for reg) + temperature/ECE.
P5 quantification: SNV/SG/arPLS, CARS/VIP, C-Mixup, calibration transfer (PDS,
   leave-one-particle-size-out).
P6 interpretability (integrated gradients peak attribution).
