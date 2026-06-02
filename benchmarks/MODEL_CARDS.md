# Model cards

Per-model configuration, intended use, metrics, and limitations for every model
in the benchmarks. Numbers come from the runner scripts on this machine (CNNs on
an RTX 4090, CUDA). Re-running regenerates them.

---

## Classification (bacteria-ID, 30 isolates)

Evaluation is **domain-shift-aware** (`scripts/run_domain_shift.py`). The
`reference` set and the `finetune`/`test` sets are different measurement
campaigns, so we report three numbers: in-distribution (reference-internal
split), cross-domain (reference -> test), adapted (finetune -> test).

All four numbers below come from `run_domain_shift.py` (single seed):

| Model | in-distribution | cross-domain (shift) | adapted | notes |
|-------|----------------:|---------------------:|--------:|-------|
| **1D-ResNet** (ResNet-18-1D, base 32, augmentation, label smoothing 0.05) | **0.940** | 0.547 | 0.759 | best in-distribution; degrades less under shift than LogReg |
| LogisticRegression | 0.919 | 0.480 | 0.806 | strong, cheap baseline; best *adapted* (robust on the 3k finetune set) |
| LinearSVM | 0.848 | 0.445 | 0.756 | linear baseline |
| RandomForest | 0.698 | 0.290 | 0.587 | weakest on raw 1000-dim spectra |

**Verdict (classification): the 1D-ResNet is the best model in-distribution
(94.0%)**, consistent with the literature that deep CNNs beat linear models on
bacteria-ID (Ho et al. 2019: ResNet 82.2% vs LR 75.7% on their split). But **all
models collapse under campaign shift** (94% -> 56%); closing that gap needs
target-domain adaptation (training on `finetune` recovers ~76-81%). This is the
field's #1 open problem, reproduced here.

### SOTA protocol (pretrain -> fine-tune -> ensemble)

Run `scripts/run_sota_classification.py`. Pretrain an SE-ResNet on the 60k
`reference` set, fine-tune on `finetune`, test on `test` - the protocol the
published numbers use. A 5-member ensemble with augmentation:

| Model | test acc |
|-------|---------:|
| Ho et al. 2019 ResNet | 0.822 |
| SANet 2026 (benchmark arch SOTA) | 0.861 |
| **heterogeneous ensemble + TTA (this repo)** | **0.862** (single 0.842 +/- 0.005) |
| SE-ResNet ensemble 2024 (open-world SOTA) | 0.878 |

An 8-member heterogeneous ensemble (4 SE-ResNet + 4 multi-scale `MSResNet1D`)
with augmentation and test-time augmentation. We beat the seminal paper (+4 pts)
and the 2026 architecture SOTA SANet (0.861), and sit ~1.5 pts below the
open-world SOTA (0.878). We also add the trust layer none of them ship: RAPS
conformal coverage 0.938 (target 0.90), average set size **1.39/30**, ECE
0.073 -> 0.025 after temperature scaling. The single-architecture variants were
weaker: plain SE-ResNet ensemble 0.852, multi-scale-only ensemble 0.844 - the
*diversity* of the heterogeneous mix plus TTA is what cleared SANet. Self-supervised
masked-AE pretraining was implemented (`SpectralMAE`, 0.711) but does not beat the
supervised ensemble here - with 60k labelled reference spectra there is little label
scarcity for it to exploit; closing the last 1.5 pts to 0.878 remains open.

### 1D-ResNet
- **Arch:** strided/residual 1-D CNN (stem conv + 4 residual stages, adaptive
  pool, FC). ~ResNet-18 over 1000-point spectra. Augmentation: offset/slope/
  multiplicative/shift/noise on the fly; label smoothing 0.05; Adam + StepLR.
- **Intended use:** closed-set isolate ID *within one measurement campaign*.
- **Trust layer:** temperature scaling (T=1.27) cut test ECE 0.144 -> 0.046;
  RAPS conformal sets reached 0.918 coverage (target 0.90), average set size
  14.9/30 under shift (honestly wide because the shifted model is uncertain).
- **Limitations:** collapses under campaign/instrument shift without adaptation;
  closed-set (see open-set card); needs a GPU to train quickly.

### LogisticRegression / LinearSVM / RandomForest
- Multinomial LR (max_iter 300), LinearSVC (squared-hinge), RF (300 trees) on
  raw normalised spectra. Cheap, transparent baselines. LR is the most robust
  after target-domain retraining. RF underperforms on high-dim raw spectra.

---

## Quantification (polystyrene dilution series)

Target: relative `log10` concentration within each particle-size series.
Repeated 5-fold CV; SD-based augmentation inside training folds only.

| Model | R^2 (CV) | RMSE (log10) | notes |
|-------|---------:|-------------:|-------|
| **RandomForest** | **0.848** | 0.201 | best; robust to the cross-size nonlinearity |
| SVR (rbf) | 0.830 | 0.212 | close second; fast |
| 1D-CNN | 0.768 | 0.247 | improves with C-Mixup; data-starved at n=48 |
| kNN | 0.690 | 0.286 | simple distance baseline |
| PLSR | 0.682 | 0.290 | linear; trails because pooling sizes is nonlinear |
| PLSR+VIP | 0.666 | 0.297 | VIP selection; no gain *here* (see note) |
| PCR | 0.589 | 0.330 | weakest |

> **VIP caveat (verified, honest):** on the raw 48-sample CV (no augmentation)
> VIP variable selection lifts PLSR R^2 0.73 -> 0.85. But in this benchmark the
> training folds are heavily SD-augmented, which makes per-fold VIP selection
> noisier and erases the gain (0.67 vs 0.68 for plain PLSR). VIP is a real win in
> the standard chemometric (non-augmented) setting; it is not additive with the
> augmentation strategy used here.

**Verdict (quantification): nonlinear classical models (RandomForest, SVR) win
on this small dataset.** The CNN needs more data than 48 spectra (C-Mixup and
SD-augmentation help but do not overtake RF). PLSR/PCR trail because pooling
particle sizes turns the mapping nonlinear (see
`agent-memory/insights/pls-fails-cross-size-pooling.md`).

**Trust layer for quantification:** jackknife+ prediction intervals (0.95
coverage at target 0.90 after the finite-sample correction) and **guarded** PDS
calibration transfer. Averaged over 25 splits, guarded PDS (transfer only when it
improves a held-out secondary check) recovers all models under a simulated
secondary-instrument response - RandomForest 0.45 -> 0.63, SVR 0.04 -> 0.50,
PLSR 0.37 -> 0.46 - without the over-correction that drives unguarded PDS
negative on shift-robust PLSR on an unlucky single split.

---

## Cross-task takeaway

Different tasks reward different algorithms: **deep CNNs win large in-distribution
classification; nonlinear classical models win small-data quantification.** The
durable contribution is not the leaderboard but the **trust layer** (calibrated
probabilities, conformal coverage, open-set rejection, calibration transfer) and
the **domain-shift-aware evaluation**, which no maintained open-source Raman
library currently provides.
