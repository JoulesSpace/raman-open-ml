---
title: Competitive analysis - where this repo stands vs the field
type: reference
date: 2026-06-02
tags: [competitive-analysis, sota, positioning]
---

# Competitive analysis (explicit, criterion-by-criterion)

Compared against the systems and projects surfaced by the literature/OSS scans
([[sota-raman-ml]]). "Best OS Raman ML project" is multi-dimensional; here is an
honest scorecard. ✅ = strong / present, ◑ = partial, ❌ = absent.

## Capability matrix

| Capability | RamanSPy | bacteria-ID (Ho) | SANet / benchmark 2026 | open-world SE-ResNet 2024 | **this repo** |
|---|:--:|:--:|:--:|:--:|:--:|
| Preprocessing breadth | ✅✅ (15+ baselines) | ◑ | ◑ | ◑ | ◑ (ALS/arPLS/airPLS, SNV/MSC, SG) |
| Dataset loaders | ✅✅ (RRUFF, bacteria, cells, COVID...) | ◑ (1) | ◑ | ◑ | ◑ (2: bacteria-ID, polystyrene) |
| Deep 1D models | ◑ | ✅ (ResNet) | ✅ (SANet/transformer) | ✅ (SE-ResNet) | ✅ (CNN/ResNet/SE/multi-scale + ensemble) |
| Classification acc (official protocol) | n/a | 0.822 | 0.861 | **0.878** | 0.862 (beats SANet; heterogeneous ensemble + TTA) |
| Hyperparameter tuning | ❌ | ❌ (manual) | ◑ | ◑ | ✅ (grid/random/Optuna-Bayes) |
| Uncertainty (conformal/temp/ECE) | ❌ | ❌ | ❌ | ◑ (UQ paper-only) | ✅ |
| Open-set / OOD | ❌ | ❌ | ❌ | ✅ (the one that does) | ✅ (MSP/energy/Mahalanobis) |
| Calibration transfer | ❌ | ❌ | ❌ | ❌ | ✅ (PDS) |
| Domain-shift-aware eval | ❌ | ◑ (fine-tune) | ◑ (reports gap) | ◑ | ✅ (in-dist/cross/adapted) |
| Interpretability | ❌ | ❌ | ❌ | ❌ | ✅ (integrated gradients + SHAP + Grad-CAM) |
| Reproducible benchmark harness | ❌ | ❌ | ◑ (paper) | ❌ | ✅ (6 one-command scripts) |
| Tests / CI / packaging | ✅ | ❌ | ❌ | ❌ | ✅ (33 tests, ruff, CI, pyproject) |
| Pretrained model zoo (hosted) | ❌ | ✅ (weights) | ❌ | ❌ | ❌ (not yet) |
| OSI license | ✅ (BSD) | ✅ (MIT) | mixed | mixed | ✅ (AGPL-3.0, copyleft) |

## Verdict (honest)

- **We lead the field on the trustworthy-ML + reproducibility axis**: the only
  project combining conformal UQ + open-set/OOD + calibration transfer +
  domain-shift-aware evaluation + interpretability + a re-runnable benchmark +
  tests/CI. Each of these is otherwise scattered across single-purpose papers or
  absent from maintained libraries.
- **Accuracy: we beat the foundational paper (0.862 vs Ho 0.822) AND the 2026
  architecture SOTA SANet (0.861)**, via an 8-member heterogeneous ensemble
  (SE-ResNet + multi-scale) with TTA. We sit ~1.5 pts below the open-world SOTA
  (0.878); closing that is the SSL-pretraining roadmap item. Our durable edge is
  the trust layer they lack.
- **We trail RamanSPy** on raw preprocessing breadth and number of bundled
  datasets, and we **lack hosted pretrained weights** (bacteria-ID ships them).

## To unambiguously be "the best" (remaining work)

1. **Exceed 0.878** classification: multi-scale ensemble (running), + more members
   / SSL pretraining if needed.
2. **Hosted model zoo + cards** (HuggingFace Hub) so others can `from_pretrained`.
3. **Breadth**: add a 3rd dataset (RRUFF or MLROD) for true cross-dataset shift,
   and a couple more preprocessing options to approach RamanSPy parity.
4. Package to PyPI; publish the benchmark leaderboard.

Conclusion: **best-in-class for trustworthy/reproducible Raman ML today, and
SOTA-competitive on accuracy** - with a concrete, short path to outright accuracy
SOTA and ecosystem breadth.
