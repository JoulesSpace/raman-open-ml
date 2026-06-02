# CLAUDE.md - operating guide for this repository

This file is mine to maintain. Whenever something here goes stale - a path moves,
a command changes, a decision is reversed - I update it in the same change that
caused the drift. A stale CLAUDE.md is a bug.

## What this project is

Open-source Raman spectroscopy machine learning on public, openly-licensed data.
Two tasks on two datasets:

- **Classification** - which analyte/species produced a spectrum (bacteria-ID,
  30 isolates).
- **Quantification** - how much analyte is present, as regression of relative
  concentration across a dilution series (polystyrene limit-of-detection set).

The point is a clean, reproducible, head-to-head comparison of which algorithm
wins for each task. See `README.md` for the framing, `DATA_SOURCES.md` for data
provenance, and `agent-memory/` for the running record of how and why.

This project uses only public, openly-licensed data. No proprietary data is
included or used.

## Hard rules (do not violate)

1. **No "assisted by Claude" / `Co-Authored-By` trailers in commits.** The user
   owns this code. Commit as the configured git author, nothing more.
2. **Semantic / conventional commits.** `type(scope): summary`
   (`feat`, `fix`, `docs`, `build`, `chore`, `refactor`, `test`, `perf`).
   Commit in logical chunks, not one giant commit.
3. **Maintain `agent-memory/`.** It is the tracked, layered memory:
   `decisions/` (ADRs), `insights/` (gotchas), `notes/` (domain knowledge), and
   `handoffs/` (dated session state), indexed by `agent-memory/MEMORY.md`.
   Update it as work happens and update the index in the same change. See
   `agent-memory/README.md` for the format.
4. **Never commit datasets.** `data/` contents are git-ignored and re-downloaded
   by `scripts/download_data.py`. Only `.gitkeep` and `.folderinfo` are tracked.
   The datasets keep their own upstream licenses (`DATA_SOURCES.md`).
5. **Every folder carries a `.folderinfo`.** Each source directory has a
   `.folderinfo` file: a one-line plain-text description of what lives there.
   Create it in the same change that creates the folder. `scripts/folderinfo.sh`
   lints this. (Downloaded data subfolders and caches are exempt.)
6. **No em-dashes.** Never write the em-dash character anywhere: prose, commit
   messages, comments, docstrings, or figures. Use a spaced hyphen `-` or
   rewrite the sentence.
7. **Report what was verified, not assumed.** Metrics in docs come from actually
   running the scripts on this machine; note the command and caveats.

## Environment

The Python environment lives at `C:\Users\julia\environments\ml-training`
(Python 3.12: numpy, pandas, scikit-learn, scipy, torch CPU, matplotlib).

```powershell
& "C:\Users\julia\environments\ml-training\Scripts\python.exe" scripts/run_classification.py
& "C:\Users\julia\environments\ml-training\Scripts\python.exe" scripts/run_quantification.py
```

`requirements.txt` lists the portable dependency set for anyone cloning fresh.

## Repository layout

```
src/raman_ml/      preprocessing (ALS/arPLS/airPLS/ModPoly/IModPoly/SNIP/
                   rubberband, cosmic-ray removal, SNV/MSC, SG deriv, norms),
                   datasets, models (sklearn + 1D-CNN + 1D-ResNet/SE/multi-scale,
                   ensembles, C-Mixup, MC-dropout), augment (SpecAugment + Raman),
                   uncertainty (conformal, temperature, ECE), ood
                   (MSP/energy/Mahalanobis), calibration_transfer (PDS),
                   interpretability (integrated grads + SHAP + Grad-CAM),
                   variable_selection (VIP), dimensionality_reduction
                   (PCA/t-SNE/UMAP/MDS/Isomap/LDA), tuning (grid/random/Optuna),
                   peaks (peak-feature extraction), ssl (masked-AE pretraining),
                   generative (conditional 1D WGAN-GP + DDPM diffusion + baselines)
scripts/           download_data, run_classification, run_quantification,
                   run_domain_shift, run_sota_classification, run_openset,
                   run_calibration_transfer, run_pca_explore, run_dimreduction,
                   run_interpretability, run_shap_overview, run_ssl_classification, run_pipeline,
                   run_generative_augmentation, plot_comparison,
                   plot_preprocessing_showcase, infographic, folderinfo (lint)
benchmarks/        results/ (CSV metrics), plots/ (PNG), MODEL_CARDS.md,
                   PLOTS_AND_METRICS.md, README
data/              datasets - git-ignored contents (.gitkeep + .folderinfo only)
assets/            generated figures for the README (raman_ml_overview.png)
agent-memory/      tracked agent memory; start at agent-memory/MEMORY.md
```

The repo's identity is the **trustworthy-ML + reproducibility layer** for Raman
(UQ, open-set/OOD, calibration transfer, domain-shift-aware eval) - the gap no
maintained OSS Raman library fills. See `agent-memory/decisions/0004` and `0005`.

## Conventions

- The two runner scripts are the cheapest oracle: re-running regenerates every
  number and plot under `benchmarks/`.
- Quantification target is **relative** concentration within each dilution
  series (single-analyte calibration), which removes the inter-particle-size
  confound. See `agent-memory/decisions/` for why.
- Preprocessing is framework-agnostic NumPy so both pipelines share it.

## When unsure

Read `agent-memory/MEMORY.md` first - it indexes everything and links the latest
handoff (current state + next steps), the decisions, and the insights.
