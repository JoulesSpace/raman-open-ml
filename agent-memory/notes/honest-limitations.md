---
title: Honest limitations - read before quoting numbers
type: note
date: 2026-06-02
tags: [limitations, honesty, evaluation]
---

# Honest limitations

What is real, what is demonstrated-on-a-proxy, and what is not yet done. Quote
numbers with these caveats.

## Solid / verified
- Classification in-distribution (1D-ResNet 0.941, LogReg 0.919) and the
  reference->test domain-shift collapse are real, on the public bacteria-ID data.
- Quantification CV numbers (RF 0.85, SVR 0.83, ...) are real on the 48-spectrum
  polystyrene set with the relative-concentration target.
- Temperature scaling + RAPS conformal coverage (0.923 at target 0.90) on the
  shifted classifier are real and reproduce on re-run.
- The test suite (14 fast unit tests) exercises every trust-layer module.

## Demonstrated on a proxy (clearly labelled)
- **Calibration transfer**: the polystyrene set is single-instrument, so the
  "secondary instrument" is a SIMULATED response (peak broadening + axis warp +
  gain). PDS genuinely recovers SVR and RF under it, but this is a controlled
  illustration, not a real two-instrument transfer. PLSR is shift-robust here and
  PDS can over-correct it.

## Context-dependent (do not over-quote)
- **VIP variable selection**: lifts PLSR R2 0.73 -> 0.85 on the **raw 48-sample
  CV (no augmentation)**, but is **neutral (0.67 vs 0.68)** in the main benchmark
  whose training folds are SD-augmented (augmentation makes per-fold VIP noisier).
- **Hyperparameter tuning**: lifts SVR 0.59 -> 0.72 and RF 0.62 -> 0.71 in a
  **no-augmentation** comparison. The benchmark instead uses SD-augmentation,
  which already lifts SVR to 0.83 - tuning and augmentation are alternative knobs
  here, not additive. Tuning was not re-run inside the augmented benchmark.

## Weak / honest negatives
- **Open-set AUROC ~0.73-0.75** only. Unknown isolates are spectrally close to
  known ones; the detectors beat chance but are far from deployable. This matches
  the literature that open-set Raman is unsolved.
- **Quantification jackknife+ coverage**: previously undershot (~0.75) due to a
  missing finite-sample level correction; fixed (now ~0.93 at target 0.90 on
  synthetic checks). Residual caveat: n=48 with grouped (per-size) structure
  still mildly violates plain exchangeability - group-aware / Mondrian conformal
  would be the rigorous next step.
- **RandomForest beats the CNN on quantification** only because n=48 is tiny;
  this is not evidence CNNs are worse for quantification in general.

## Not yet done
- No SSL pretraining, no GAN/diffusion augmentation, no CARS/VIP selection.
- No real multi-instrument dataset; no cross-dataset (RRUFF/MLROD) eval.
- Deep ensembles and Integrated-Gradients interpretability now exist as modules
  (`models.DeepEnsemble`, `interpretability.py`) but are not yet wired into the
  benchmark scripts / leaderboard. No PyPI release yet.
- bacteria-ID "adapted" ResNet (0.759) underperforms adapted LogReg (0.806):
  the ResNet overfits the 3k finetune set; needs SSL pretraining or the ensemble
  applied to the adaptation step.
