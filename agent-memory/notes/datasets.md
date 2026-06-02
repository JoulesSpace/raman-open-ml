---
title: Dataset notes (bacteria-ID, polystyrene)
type: note
date: 2026-06-02
tags: [data]
---

# Dataset notes

Full provenance and licensing live in `../../DATA_SOURCES.md`. This note is the
practical shape-and-quirks reference.

## bacteria-ID (classification)

- Source arrays on a Dropbox folder; `?dl=1` returns a ~628 MB zip of `.npy`
  files. `download_data.py` fetches and extracts it.
- Splits: `reference` (60000, 1000), `finetune` (3000), `test` (3000), plus
  `2018clinical` / `2019clinical`. Labels are int 0..29.
- Wavenumbers: 1000 points, **descending** 1792.40 -> 381.98 cm^-1.
- Already baseline-corrected and min-max [0, 1] scaled; we use them as-is.
- 30 classes (`datasets.STRAINS`) span yeast (Candida), Gram-negative and
  Gram-positive bacteria, MSSA/MRSA, and several Strep groups. 30-class isolate
  ID is genuinely hard; the upstream paper reports ~82% with a deep ResNet.

## polystyrene limit-of-detection (quantification)

- Renishaw inVia, 785 nm, 100% power, 10 s, 5X, 1200 l/mm grating.
- `Fig.S1..S8` = 8 particle sizes (25..1000 nm). Each file: 10 column-blocks of
  [Raman Shift, mean Intensity, SD]; 6 of the blocks are concentrations, the rest
  are controls (96-well plate, Water, NaNO3, Tween 20).
- Each size is a halving series; the 8 series together span
  6.56e8 .. 7.08e14 particles/mL.
- We interpolate every spectrum onto a common 1000-point grid
  (~399.5 .. 3202.7 cm^-1) and carry the SD for noise-based augmentation.
- Tiny (48 mean spectra). Treat with small-sample discipline: repeated K-fold CV,
  augment training folds only, never the test fold.
