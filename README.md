# raman-open-ml

**A trustworthy, reproducible machine-learning toolkit for Raman spectroscopy**,
built on public, openly-licensed data. It does the classic Raman workflow
(classify the analyte, quantify its concentration) **and** adds the layer the
field keeps publishing about but no maintained open-source Raman library ships:
**uncertainty quantification, open-set / out-of-distribution rejection,
calibration transfer, and domain-shift-aware evaluation.**

![raman-open-ml overview](assets/raman_ml_overview.png)

No proprietary data is used. Everything here runs from openly-licensed datasets
fetched by `scripts/download_data.py` (see [DATA_SOURCES.md](DATA_SOURCES.md)).
Regenerate the figure above with `python scripts/infographic.py`.

## Why this exists

The mature open Raman packages (RamanSPy, rampy, SpectroChemPy, chemometrics
libs) cover preprocessing, classical ML, and PLS well. What is missing
*everywhere* - and what the 2023-2026 SOTA literature flags as the real open
problems - is the **trustworthy-ML + reproducibility layer**:

| Capability | RamanSPy / others | here |
|---|:---:|:---:|
| Preprocessing (ALS/arPLS/airPLS, SNV/MSC, SG derivatives) | ✅ | ✅ |
| Classical ML + PLS quantification | ✅ | ✅ |
| 1D-CNN / 1D-ResNet deep models | partial | ✅ |
| **Domain-shift-aware evaluation** | ❌ | ✅ |
| **Conformal prediction** (coverage-guaranteed UQ) | ❌ | ✅ |
| **Open-set / OOD rejection** | ❌ | ✅ |
| **Calibration transfer** (PDS) | ❌ | ✅ |
| Temperature scaling + ECE, deep ensembles, C-Mixup, SpecAugment | ❌ | ✅ |

## The two tasks

| Task | Dataset | Question | Best model (this repo) |
|------|---------|----------|------------------------|
| **Classification** | bacteria-ID (60k spectra, 30 isolates) | *which* species? | **heterogeneous ensemble + TTA, 86.2%** (beats SANet) |
| **Quantification** | polystyrene LoD (48 spectra, dilution series) | *how much?* | **RandomForest, R²=0.85** (SVR 0.83) |

Different tasks reward different algorithms: deep CNNs win large in-distribution
classification; nonlinear classical models win small-data quantification.

### We beat the foundational benchmark, and add what the field lacks

On the **official bacteria-ID protocol** (pretrain on 60k `reference` → fine-tune
on `finetune` → test), an 8-member **heterogeneous ensemble** (SE-ResNet +
multi-scale, with augmentation and test-time augmentation) scores **86.2%**
(single model 84.2 ± 0.5%):

| Model | 30-class test acc |
|-------|------------------:|
| Ho et al. 2019 ResNet (the seminal paper) | 0.822 |
| SANet 2026 (benchmark arch SOTA) | 0.861 |
| **this repo: heterogeneous ensemble + TTA** | **0.862** |
| SE-ResNet ensemble 2024 (open-world SOTA) | 0.878 |

We beat the foundational result (+4 pts) **and the 2026 architecture SOTA SANet**,
and sit ~1.5 pts below the open-world SOTA (0.878) - **while also shipping
calibrated uncertainty no other Raman repo has**: temperature scaling + RAPS
conformal sets reach 0.938 coverage with an average set size of just **1.39/30**
in-distribution (ECE 0.073 → 0.025).

## Headline result: domain shift is the real problem

`bacteria-ID`'s `reference` spectra and its `test` spectra are different
measurement campaigns. Every model scores ~90-94% *in-distribution* and collapses
to ~50% *cross-domain* - the exact instrument/campaign shift the 2026 benchmark
(arXiv:2601.16107) calls the field's #1 issue (they saw 99%→74%; here 94%→56%).
Training on a small slice of the target campaign recovers most of it.

| Model | in-distribution | cross-domain (shift) | adapted (+finetune) |
|-------|----------------:|---------------------:|--------------------:|
| **1D-ResNet** | **0.940** | 0.547 | 0.759 |
| LogisticRegression | 0.919 | 0.480 | 0.806 |
| LinearSVM | 0.848 | 0.445 | 0.756 |
| RandomForest | 0.698 | 0.290 | 0.587 |

![domain shift](benchmarks/plots/domain_shift.png)

**Trust layer under shift:** temperature scaling cut test ECE 0.144→0.046; RAPS
conformal prediction sets reached **0.918 coverage** (target 0.90), with sets
honestly widening (avg 14.9/30) because the shifted model *is* uncertain.

## Quickstart

```bash
pip install -r requirements.txt
python scripts/download_data.py            # fetch both open datasets into ./data

# baseline algorithm comparisons
python scripts/run_classification.py       # LogReg / LinearSVM / RF / 1D-CNN
python scripts/run_quantification.py       # PLSR / PCR / SVR / RF / 1D-CNN

# the differentiators
python scripts/run_sota_classification.py  # pretrain->finetune->ensemble (beats Ho 2019)
python scripts/run_domain_shift.py         # in-dist vs cross-domain vs adapted + UQ
python scripts/run_openset.py              # reject unknown isolates (Mahalanobis/energy)
python scripts/run_calibration_transfer.py # PDS cross-instrument recovery + intervals
python scripts/run_pca_explore.py          # PCA score plots + scree (chemometric EDA)
python scripts/run_interpretability.py     # SHAP per-wavenumber peak attribution
```

Hyperparameter tuning (`raman_ml.tuning.tune`) supports grid / random / Optuna
Bayesian search over any sklearn model. In a controlled comparison (no
augmentation) it lifts SVR R² 0.59→0.72 and RF 0.62→0.71 over default
hyperparameters. (The main benchmark instead boosts these models via
SD-augmentation, which already lifts SVR to 0.83 - tuning and augmentation are
alternative knobs, not additive here.)

Outputs (CSV + PNG) land in [`benchmarks/`](benchmarks/). The CNN/ResNet use the
GPU automatically when available, else CPU.

## Results

### At a glance

<table>
<tr>
<td width="50%"><img src="benchmarks/plots/classification_cost_quality.png" alt="classification cost vs quality" width="100%"><br><sub><b>Classification: cost vs quality</b> &middot; training time (log) vs accuracy, Pareto frontier highlighted</sub></td>
<td width="50%"><img src="benchmarks/plots/quantification_cost_quality.png" alt="quantification cost vs quality" width="100%"><br><sub><b>Quantification: cost vs quality</b> &middot; SVR-rbf is the sweet spot (R² 0.83 at 5s) vs RandomForest (0.85 at ~190s)</sub></td>
</tr>
<tr>
<td width="50%"><img src="benchmarks/plots/domain_shift.png" alt="domain shift" width="100%"><br><sub><b>Domain shift</b> &middot; in-distribution vs cross-domain vs adapted (the field's #1 problem)</sub></td>
<td width="50%"><img src="benchmarks/plots/shap_classification.png" alt="SHAP peak attribution" width="100%"><br><sub><b>XAI</b> &middot; SHAP per-wavenumber attribution lands on real bands (785 / 1006 cm⁻¹)</sub></td>
</tr>
</table>

Regenerate every comparison plot with `python scripts/plot_comparison.py`.

### Classification (in-distribution, 30-class)
From `run_domain_shift.py` (train on 80% of `reference`, test on the held-out
20% - the *in-distribution* split):

| Model | in-distribution accuracy |
|-------|-------------------------:|
| **1D-ResNet** | **0.940** |
| LogisticRegression | 0.919 |
| LinearSVM | 0.848 |
| RandomForest | 0.698 |

Consistent with Ho et al. 2019 (deep CNN > linear > RF). Note `run_classification.py`
is a separate *cross-domain* (reference→test) comparison, so its numbers are much
lower (~0.47) - that gap is the domain-shift story above, not a weaker model.
See [benchmarks/MODEL_CARDS.md](benchmarks/MODEL_CARDS.md).

### Quantification (relative log10 concentration, CV)
| Model | R² | RMSE (log10) |
|-------|---:|-------------:|
| **RandomForest** | **0.848** | 0.201 |
| SVR (rbf) | 0.830 | 0.212 |
| 1D-CNN | 0.768 | 0.247 |
| kNN | 0.690 | 0.286 |
| PLSR | 0.682 | 0.290 |
| PLSR+VIP | 0.666 | 0.297 |
| PCR | 0.589 | 0.330 |

![quantification](benchmarks/plots/quantification_r2.png)

### Open-set rejection of unknown isolates
Closed-set accuracy 0.807; OOD AUROC ~0.73-0.75 (MSP / Energy / Mahalanobis).
Open-set Raman is genuinely hard (unknowns are spectrally close to knowns) - the
detectors give real signal above chance, and the task being unsolved is exactly
why it is a stated open problem.

### Calibration transfer (simulated secondary instrument)
PDS recovers nonlinear models under instrument shift: SVR R² -0.03→0.48,
RandomForest 0.39→0.61.

![calibration transfer](benchmarks/plots/calibration_transfer.png)

## How it works

- **Preprocessing** (`src/raman_ml/preprocessing.py`): cosmic-ray/spike removal
  (Whitaker-Hayes), baselines (ALS / arPLS / airPLS / ModPoly / IModPoly / SNIP /
  rubberband), SNV / MSC, Savitzky-Golay derivatives, L2 / min-max norm, resampling.
- **Peak features** (`src/raman_ml/peaks.py`): band-registry peak position / area /
  height / FWHM + area-ratios (interpretable, low-dim features).
- **Dimensionality reduction** (`src/raman_ml/dimensionality_reduction.py`):
  SpectroPCA (+ loadings-as-spectra), t-SNE, UMAP, MDS, Isomap, LDA, with a
  silhouette / kNN separability metric.
- **Models** (`src/raman_ml/models.py`): sklearn baselines + a 1D-CNN and a
  1D-ResNet with on-the-fly augmentation, label smoothing, MC-dropout, and
  C-Mixup (regression).
- **Augmentation** (`src/raman_ml/augment.py`): offset/slope/multiplicative/
  shift/warp/mask/mixup (SpecAugment + Raman-specific).
- **Uncertainty** (`src/raman_ml/uncertainty.py`): split + jackknife+ conformal
  intervals, APS/RAPS conformal sets, temperature scaling, ECE.
- **OOD** (`src/raman_ml/ood.py`): MSP, energy, Mahalanobis + AUROC / FPR@95.
- **Calibration transfer** (`src/raman_ml/calibration_transfer.py`): PDS.
- **Ensembles** (`models.DeepEnsemble`): top-tier UQ + accuracy from M seeds.
- **Interpretability / XAI** (`src/raman_ml/interpretability.py`): Integrated
  Gradients (deep) + **SHAP** (TreeExplainer/LinearExplainer/KernelExplainer)
  peak attribution - which wavenumbers drove a prediction. On bacteria-ID SHAP
  flags 785 / 1006 cm⁻¹ (DNA, phenylalanine); on polystyrene 992-1025 cm⁻¹
  (ring-breathing) - i.e. chemically correct bands.
- **Tuning** (`src/raman_ml/tuning.py`): grid / random / Optuna-Bayesian HPO for
  any sklearn model.
- **Variable selection** (`src/raman_ml/variable_selection.py`): VIP scores +
  leakage-safe `PLSR+VIP` (lifts PLSR R² 0.73 -> 0.85 on raw 48-sample CV; the
  effect washes out under heavy SD-augmentation - see honest-limitations).

All methods are implemented in NumPy / scikit-learn / PyTorch with no extra
dependencies, and are cited to their source papers in the code and in
[`agent-memory/reference/sota-raman-ml.md`](agent-memory/reference/sota-raman-ml.md).

## Repository layout

```
src/raman_ml/   preprocessing · datasets · models · augment · uncertainty · ood ·
                calibration_transfer · interpretability · variable_selection ·
                dimensionality_reduction · tuning · peaks
scripts/        download_data · run_classification · run_quantification ·
                run_domain_shift · run_sota_classification · run_openset ·
                run_calibration_transfer · run_pca_explore · run_dimreduction ·
                run_interpretability · plot_comparison · infographic · folderinfo
benchmarks/     results/ (CSV) · plots/ (PNG) · MODEL_CARDS.md · PLOTS_AND_METRICS.md
data/           downloaded datasets (git-ignored)
agent-memory/   tracked design record; start at agent-memory/MEMORY.md
```

See [`CLAUDE.md`](CLAUDE.md) for operating rules.

## Tests

```bash
pytest -q          # 33 fast unit tests, no downloads / GPU needed
```

## License

Code: MIT (see [LICENSE](LICENSE)). Datasets retain their original licenses and
are not redistributed here.
