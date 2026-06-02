"""Can augmentation push the top-5 unified-pipeline configs past R2 = 0.792?

The unified sweep (run_pipeline.py) ranks 96 pipelines with NO augmentation; the
winner is ALS + L2 + SG-d1 + RandomForest at R2 = 0.792 (mean per-fold, CV). This
script re-runs the top-5 configs with augmentation added to the TRAINING folds
only (leakage-safe), using the *same* preprocess/build/CV as the sweep so the
numbers are directly comparable.

Note on the augmenter: the quantification set is 48 mean spectra, far too few to
train a WGAN-GP/DDPM (they would collapse) - so the appropriate generative model
here is the dataset's own measurement-noise model: sample replicates
~ N(mean, measured per-point SD). That is the principled "generative augmentation"
for replicate spectra, and it is what the classification GAN approximates anyway.

    python scripts/run_pipeline_augmented.py --n-aug 40

Outputs:
  results/pipeline_augmented_metrics.csv
  plots/pipeline_augmented.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))

from run_pipeline import build, preprocess  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import RepeatedKFold  # noqa: E402

from raman_ml.datasets import load_polystyrene  # noqa: E402

_ROOT = os.path.dirname(_HERE)
RESULTS_DIR = os.path.join(_ROOT, "benchmarks", "results")
PLOTS_DIR = os.path.join(_ROOT, "benchmarks", "plots")


def _augment_raw(X, SD, y, n_aug, rng):
    """Replicate each training spectrum n_aug times with N(0, SD) measurement noise."""
    Xs, ys = [X], [y]
    for _ in range(n_aug):
        Xs.append(X + rng.standard_normal(X.shape) * SD)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def _cv_r2(X_raw, SD_raw, y, cfg, cv, n_aug, seed):
    """Mean per-fold R2 (matches the sweep); augment training folds only."""
    bl, nm, sg, dr, mdl = cfg
    rng = np.random.default_rng(seed)
    noaug, aug = [], []
    for tr, te in cv.split(X_raw):
        Xp_te = preprocess(X_raw[te], bl, nm, sg)
        # no augmentation
        est = build(dr, mdl).fit(preprocess(X_raw[tr], bl, nm, sg), y[tr])
        noaug.append(r2_score(y[te], est.predict(Xp_te)))
        # SD-noise augmentation on the training fold
        Xtr_a, ytr_a = _augment_raw(X_raw[tr], SD_raw[tr], y[tr], n_aug, rng)
        est = build(dr, mdl).fit(preprocess(Xtr_a, bl, nm, sg), ytr_a)
        aug.append(r2_score(y[te], est.predict(Xp_te)))
    return float(np.mean(noaug)), float(np.mean(aug))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-aug", type=int, default=40)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    X_raw, SD_raw, conc, size_id, _ = load_polystyrene(1000)
    ylog = np.log10(conc); y = ylog.copy()
    for s in np.unique(size_id):
        y[size_id == s] = ylog[size_id == s] - ylog[size_id == s].max()
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=args.seed)

    sweep = pd.read_csv(os.path.join(RESULTS_DIR, "pipeline_sweep.csv")).head(args.top)
    rows = []
    for r in sweep.itertuples():
        cfg = (r.baseline, r.norm, r.sg, r.dr, r.model)
        base, aug = _cv_r2(X_raw, SD_raw, y, cfg, cv, args.n_aug, args.seed)
        label = f"{r.model}|{r.baseline}|{r.norm}|{r.sg}|{r.dr}"
        rows.append(dict(pipeline=label, r2_no_aug=base, r2_augmented=aug,
                         delta=aug - base))
        print(f"  {label:24s} no-aug={base:.3f}  +SD-aug={aug:.3f}  (delta {aug-base:+.3f})")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "pipeline_augmented_metrics.csv"), index=False)
    best = df["r2_augmented"].max()
    print(f"\nBest augmented R2 = {best:.3f}  (unaugmented sweep winner = 0.792, "
          f"HPO 0.808)  ->  {'EXCEEDS' if best > 0.792 else 'does not exceed'} 0.792")

    # grouped bar: no-aug vs +SD-aug per pipeline
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(df)); w = 0.38
    ax.bar(x - w / 2, df["r2_no_aug"], w, label="no augmentation", color="#9aa3b2")
    ax.bar(x + w / 2, df["r2_augmented"], w, label="+ SD-noise augmentation", color="#2a9d8f")
    ax.axhline(0.792, color="#12263a", ls=":", lw=1, label="sweep winner 0.792")
    ax.set_xticks(x); ax.set_xticklabels(df["pipeline"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("R$^2$ (mean per-fold CV)"); ax.set_ylim(0, 1)
    ax.set_title("Top-5 unified pipelines: does training-fold augmentation beat 0.792?")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS_DIR, "pipeline_augmented.png"), dpi=130)
    print("\nSaved pipeline_augmented_metrics.csv + pipeline_augmented.png")


if __name__ == "__main__":
    main()
