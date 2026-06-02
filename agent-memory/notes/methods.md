---
title: Method notes (preprocessing, models, evaluation)
type: note
date: 2026-06-02
tags: [preprocessing, models, evaluation]
---

# Method notes

## Preprocessing (`src/raman_ml/preprocessing.py`)

- **ALS baseline** (Eilers & Boelens): `lam=1e5`, `p=0.01`, 10 iterations.
  Build the difference operator with float diagonals and solve with a CSC matrix
  (avoids a dtype-cast warning and a spsolve format warning).
- **Normalisation:** L2 (default), SNV, or min-max, all per-spectrum.
- **Resampling:** `np.interp` onto a common grid, order-agnostic (handles the
  descending bacteria axis and the ascending polystyrene axis).
- bacteria-ID arrives preprocessed, so its runner uses the spectra as-is;
  polystyrene is baseline-removed + L2-normalised from raw.

## Models (`src/raman_ml/models.py`)

- Classical: sklearn LogisticRegression, LinearSVC, RandomForest (clf) and
  PLSR, PCR, SVR-rbf, RandomForest (reg).
- `SpectralCNN`: 3 conv blocks (conv-BN-ReLU-maxpool) -> adaptive pool -> MLP
  head. `pool_out` trades spatial detail (clf) vs regularisation (reg).
- CNN wrappers expose `fit`/`predict`; the regressor standardises its target
  (see `../insights/cnn-regressor-target-standardization.md`). Adam + StepLR.
- Device resolves to CUDA when available.

## Evaluation

- **Classification:** shared stratified train subsample, score on the full 3k
  test split (accuracy, balanced accuracy, macro-F1). Confusion matrix for the
  winner.
- **Quantification:** RepeatedKFold on the 48 real spectra; inside each train
  fold, add `n_aug` Gaussian replicates drawn from the per-point SD (scaled into
  the L2-normalised domain by dividing by the spectrum's norm). Test folds stay
  real. Report pooled R2 / RMSE / MAE in log units plus per-fold mean +/- std.

## Augmentation rationale

The SD columns are the empirical spread across replicate acquisitions, so
`N(mean, SD)` replicates are physically grounded measurement noise, not invented
structure. This is the small-sample analogue of the GAN augmentation used in the
proprietary GAN-augmentation pipelines, but transparent and leakage-free.
