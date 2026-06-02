# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/); versions follow semver.

## [Unreleased]

### Added
- Trustworthy-ML layer: conformal prediction (split + jackknife+ intervals,
  APS/RAPS sets), temperature scaling + ECE, deep ensembles, MC-dropout.
- Open-set / OOD detection: MSP, energy, Mahalanobis (+ AUROC / FPR@95).
- Calibration transfer (Piecewise Direct Standardization).
- Domain-shift-aware classification evaluation (in-distribution / cross-domain /
  adapted) and the SOTA-protocol ensemble runner.
- Models: 1D-CNN, 1D-ResNet, SE attention, multi-scale `MSResNet1D`, heterogeneous
  deep ensembles, C-Mixup; transfer-learning `finetune`; `save`/`load_cnn`
  persistence; `set_global_determinism`.
- Preprocessing: ALS/arPLS/airPLS/ModPoly/IModPoly/SNIP/rubberband baselines,
  cosmic-ray removal (Whitaker-Hayes), SNV/MSC, Savitzky-Golay derivatives.
- Dimensionality reduction: PCA/SpectroPCA, t-SNE, UMAP, MDS, Isomap, LDA with a
  separability metric.
- Interpretability: Integrated Gradients, SHAP, Grad-CAM (per-wavenumber).
- Hyperparameter tuning (grid / random / Optuna Bayesian); VIP variable
  selection; peak-feature extraction.
- Tooling: pytest suite, ruff, pre-commit, GitHub Actions CI, Makefile,
  hero infographic + cost-vs-quality comparison plots.

### Fixed
- Mahalanobis OOD: pooled within-class covariance now uses the correct
  (N - K) estimator instead of `np.cov` (which re-centred deviations).
- jackknife+ intervals: added the finite-sample level correction; coverage on
  the small quantification set rose from ~0.75 to ~0.95 at a 0.90 target.
- CNN regressor target standardisation (prevented divergence on log-conc targets).

### Changed
- License: MIT -> **AGPL-3.0-or-later** (strong copyleft, network-use clause).

### Notes
- Best classification (official bacteria-ID protocol): heterogeneous ensemble +
  TTA = 0.862 (beats Ho 2019 0.822 and SANet 2026 0.861; below open-world 0.878).
