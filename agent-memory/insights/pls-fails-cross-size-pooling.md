---
title: Linear PLS/PCR fail when pooling particle sizes; nonlinear models cope
type: insight
date: 2026-06-02
tags: [quantification, pls, confound]
---

# Pooling particle sizes makes absolute quantification nonlinear

Within a single particle size, total Raman intensity tracks log concentration
almost perfectly (r = 0.86 .. 0.98 for 7 of 8 sizes; size index 1 is an outlier
at -0.18). But across sizes the same intensity maps to different absolute
particle counts, so a global linear model cannot fit the pooled set.

**Verified (no augmentation, 5-fold CV, absolute target):**
- PLSR: R2 = -3 to -4.6 across 2..15 components
- SVR-rbf: R2 = 0.59

The nonlinear models (RF, SVR-rbf) partly recover the multi-cluster structure;
the linear ones (PLSR, PCR) cannot. Two fixes, both applied:
1. Switch to the **relative** target (decision 0002) - removes the confound.
2. Keep nonlinear models in the lineup - they remain the winners even on the
   relative target (RF 0.85, SVR 0.83 vs PLSR 0.68).

Takeaway: "PLSR is the Raman quantification default" is only true for a single
consistent calibration. Mixing experimental conditions (here, particle size)
silently turns the problem nonlinear.
