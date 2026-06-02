---
title: Position the repo as the trustworthy-ML + reproducibility layer
type: decision
date: 2026-06-02
status: accepted
tags: [positioning, uncertainty, ood, calibration-transfer]
---

# 0004 - Own the trustworthy-ML layer, do not re-implement RamanSPy

## Context

A literature + OSS scan (see [[sota-raman-ml]]) showed RamanSPy already owns
preprocessing breadth, dataset loaders, and classical ML; rampy/SpectroChemPy/
chemometrics libs cover PLS/peak-fitting. Re-implementing those adds no value.
The 2023-2026 SOTA papers repeatedly flag the same unmet needs - open-set/OOD,
domain/instrument shift + calibration transfer, uncertainty, reproducibility -
and none of these are in any maintained Raman library.

## Decision

Build the **trustworthy-ML + reproducibility layer** as the repo's identity:

- `uncertainty.py`: split + jackknife+ conformal intervals, APS/RAPS conformal
  sets, temperature scaling, ECE.
- `ood.py`: MSP, energy, Mahalanobis detectors + AUROC / FPR@95.
- `calibration_transfer.py`: PDS.
- `augment.py`: SpecAugment + Raman-specific augmentation; C-Mixup in the CNN.
- `models.py`: 1D-ResNet (SOTA arch) alongside the compact CNN.
- domain-shift-aware evaluation (see 0005).

All implemented in numpy/sklearn/torch with **no new dependencies**, each cited
to its source paper.

## Consequences

- The "best algorithm per task" leaderboard remains, but the durable
  contribution is the trust layer + honest evaluation.
- We deliberately do NOT vendor RamanSPy's 15 baseline algorithms; we add the
  few that matter (arPLS/airPLS/SNV/MSC/SG) and point to RamanSPy for breadth.
