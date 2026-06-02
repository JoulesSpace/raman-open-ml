---
title: CNN regressor diverges unless the target is standardised
type: insight
date: 2026-06-02
tags: [cnn, regression, gotcha]
---

# CNN regressor needs target standardisation

The quantification target is `log10` concentration, originally in the range
~9 .. 15 (absolute) and ~ -1.5 .. 0 (relative). A freshly-initialised network
outputs values near 0, so MSE against a target far from 0 produces huge initial
gradients and the regressor diverges.

**Verified:** without standardisation the CNN regressor scored R2 = -53 (absolute
target). Z-scoring the target inside `CNNRegressor.fit` (store mean/std, train on
the z-scored target, invert at predict) fixed it; the CNN then lands at a sensible
R2 ~ 0.75 on the relative target.

Takeaway: any neural regressor on spectroscopic concentration targets should
standardise `y`. The sklearn models are scale-robust and did not need this, which
is exactly why the bug was invisible until the CNN was added.
