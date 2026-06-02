---
title: Quantification target is relative concentration within each dilution series
type: decision
date: 2026-06-02
status: accepted
tags: [quantification, target, confound]
---

# 0002 - Relative-concentration target for quantification

## Context

The polystyrene set has 8 particle sizes, each measured as its own halving
dilution series. The obvious target is absolute `log10(particles/mL)`. But the
same particle-number concentration corresponds to very different amounts of
polystyrene (and thus very different Raman signal) across sizes, so absolute
concentration is not recoverable from a single spectrum when the size is unknown.

Verified: pooling all sizes and predicting absolute log10 concentration gives
strongly negative CV R2 for the linear chemometric models (PLSR around -0.4 with
augmentation, worse without), while within each size the correlation between
total intensity and log concentration is r = 0.86 .. 0.98 (one outlier size at
-0.18). See `insights/pls-fails-cross-size-pooling.md`.

## Decision

Define the target as the **relative** concentration within each size's series:
`y = log10(conc) - log10(max conc of that size)`. This is the single-analyte
calibration framing (recover the dilution level / relative analyte amount from
spectral intensity), it is consistent across all 8 sizes, and it matches the
"one analyte across a dilution series" task the project is modelled on.

## Consequences

- All five regressors become comparable and meaningful; PLSR recovers to
  R2 ~ 0.68 and nonlinear models reach ~0.83 - 0.85.
- The absolute-concentration caveat is documented in `DATA_SOURCES.md`; the
  parity plot is colour-coded by particle size so residual size structure is
  visible.
- Default PLS components dropped from 10 to 3 (10 overfit 48 samples).
