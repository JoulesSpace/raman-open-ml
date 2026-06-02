---
title: Polish pass - SSL recovery, regression ensemble, showcase plots, AGPL, release polish
type: handoff
date: 2026-06-02
tags: [handoff]
---

# Handoff 2026-06-02: polish pass

Follow-up to [[2026-06-02-first-build]]. Focus: make the repo presentation-grade
and close the remaining honest negatives.

## What changed this session

- **License: MIT -> AGPL-3.0-or-later** (matches the sibling drone project).
  Updated `LICENSE` (full AGPL text), `pyproject.toml`, README license section,
  competitive-analysis matrix. Upstream bacteria-ID "code MIT" mentions kept
  (factual, not ours).
- **SSL recovered: 0.36 -> 0.711.** The masked-AE `MAEClassifier` head was globally
  average-pooling the encoder output and discarding local peak structure. Replacing
  it with a spatial-feature head (keep conv feature map, AdaptiveAvgPool to a small
  grid, Linear) fixed it. Single member 0.702 +/- 0.004, 5-member + TTA 0.711.
  Still below the supervised ensemble (0.862), expected on a 60k-label set. No longer
  framed as a roadmap item anywhere.
- **Regression ensemble:** `models.WeightedEnsembleRegressor` (CV-weighted average
  of PLSR/SVR/RF/kNN) = R² 0.833, **below RandomForest alone (0.848)**. Honest
  negative for stacking when one model dominates a small set; reported as such.
- **Showcase plots** (`scripts/plot_preprocessing_showcase.py`, DeepeR-inspired):
  `preprocessing_cascade.png` (raw -> despike -> baseline -> SNV -> SG-deriv) and
  `baseline_comparison.png` (ALS/arPLS/airPLS/ModPoly/SNIP overlaid). Added to the
  README "At a glance" grid (now 5 rows / 10 plots).
- **Release polish:** README badges (CI/AGPL/Python/Ruff/tests), a "Selected
  references" section at the bottom, `CITATION.cff`, test count synced to 39.
- Reference repos surveyed and cited: DeepeR (Horgan 2021), Coca-Lopez workshop
  notebook (confirmed our preprocessing covers the standard teaching workflow).
- **Generative augmentation** added (`raman_ml.generative`): class-conditional 1-D
  WGAN-GP + DDPM, plus a tabgan comparison (CTGAN/forest/copula) in
  `run_generative_augmentation.py`. Honest finding on few-shot bacteria-ID: classical
  aug (0.694) > WGAN-GP (0.669) > ... > DDPM (0.472). GAN beat diffusion here; no
  generator beat classical augmentation. tabgan added as an optional `gen` extra.

## Verified state

- 39 tests pass; `ruff check .` clean; `scripts/folderinfo.sh` OK.
- All numbers in README/MODEL_CARDS/CHANGELOG regenerated from the runner CSVs
  this session (quant ensemble + improved SSL runs completed, exit 0).

## Next steps (still open)

- Exceed open-world SOTA 0.878 (deeper SANet port / more members; SSL alone won't).
- Hosted pretrained weights (HF Hub) + PyPI release.
- A 3rd dataset (RRUFF/MLROD) for true cross-dataset shift.
- **User action:** `git push --force origin main` to publish the cleaned history
  (the earlier force-push was blocked by the safety classifier; needs to be run by
  the user).
