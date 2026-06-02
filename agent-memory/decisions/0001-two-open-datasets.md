---
title: Use two separate open datasets (classification vs quantification)
type: decision
date: 2026-06-02
status: accepted
tags: [data, classification, quantification]
---

# 0001 - Two open datasets, one per task

## Context

The project mirrors a "classify the analyte, then quantify its concentration"
Raman workflow, but on public data only (no proprietary data).
Classification needs categorical labels; quantification needs continuous
concentration labels. No single well-known open Raman set provides both cleanly.

## Decision

Use two openly-licensed datasets:

- **Classification: bacteria-ID** (Ho et al. 2019, code MIT). 60k reference +
  3k test spectra, 30 isolates, 1000 wavenumbers, already baseline-corrected and
  [0, 1] scaled. Large and proven with CNNs.
- **Quantification: polystyrene limit-of-detection** (Mendeley 33wf5rtr4h, CC BY
  4.0). 8 particle sizes, each a 6-point halving dilution series, 48 mean spectra
  with per-wavenumber SD, spanning 6.56e8 .. 7.08e14 particles/mL.

Both are fetched by `scripts/download_data.py`; neither is committed.

## Consequences

- The two tasks live behind one shared preprocessing + model layer but separate
  runner scripts.
- The size mismatch (60k vs 48) is itself instructive: it is the main reason the
  best algorithm differs by task (see 0002 and the insights).

## Considered and rejected

- **Viral SERS dataset (Mendeley 44sgp2jvj5, CC BY 4.0):** its files are
  data-behind-figures; `Fig_3_data.csv` held only 2 example spectra, not a
  trainable matrix. See `insights/mendeley-figure-csvs.md`.
