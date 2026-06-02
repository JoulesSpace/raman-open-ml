"""Classification benchmark on the bacteria-ID dataset.

Trains several algorithms on a shared training subsample and evaluates them on
the held-out test split, then writes a metrics table and plots to ./results.

    python scripts/run_classification.py --train-size 20000 --cnn-epochs 20
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.svm import LinearSVC

from raman_ml.datasets import STRAINS, load_bacteria_id
from raman_ml.models import CNNClassifier

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "benchmarks", "results")
PLOTS_DIR = os.path.join(_ROOT, "benchmarks", "plots")


def model_factories(args):
    """Seed-parameterised factories so each model can be trained over several
    seeds for a reproducibility (seed-variance) error bar."""
    return {
        "LogisticRegression": lambda s: LogisticRegression(max_iter=300, C=1.0),
        "LinearSVM": lambda s: LinearSVC(C=1.0, dual=False, random_state=s),
        "RandomForest": lambda s: RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=s),
        "1D-CNN": lambda s: CNNClassifier(
            n_out=30, epochs=args.cnn_epochs, batch_size=256,
            channels=(64, 128, 256), pool_out=8, seed=s, verbose=args.verbose),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=20000)
    ap.add_argument("--cnn-epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3,
                    help="number of random seeds per model (seed-variance bars)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Loading bacteria-ID (train subsample={args.train_size}) ...")
    X_tr, y_tr, wn = load_bacteria_id("reference", subsample=args.train_size,
                                      seed=args.seed)
    X_te, y_te, _ = load_bacteria_id("test")
    print(f"  train={X_tr.shape}  test={X_te.shape}  classes={len(STRAINS)}")

    seeds = list(range(args.seeds))
    rows = []
    best = (None, -1.0, None)
    for name, factory in model_factories(args).items():
        print(f"\n>>> {name}  ({len(seeds)} seed(s))")
        accs, bals, f1s, times = [], [], [], []
        pred0 = None
        for si, s in enumerate(seeds):
            t0 = time.time()
            model = factory(s).fit(X_tr, y_tr)
            times.append(time.time() - t0)
            pred = model.predict(X_te)
            accs.append(accuracy_score(y_te, pred))
            bals.append(balanced_accuracy_score(y_te, pred))
            f1s.append(f1_score(y_te, pred, average="macro"))
            if si == 0:
                pred0 = pred
        acc_m, acc_s = float(np.mean(accs)), float(np.std(accs))
        print(f"    acc={acc_m:.4f}+/-{acc_s:.4f}  "
              f"balanced_acc={np.mean(bals):.4f}  macroF1={np.mean(f1s):.4f}  "
              f"fit={np.mean(times):.1f}s")
        rows.append(dict(model=name, accuracy=acc_m, accuracy_std=acc_s,
                         balanced_accuracy=float(np.mean(bals)),
                         macro_f1=float(np.mean(f1s)),
                         macro_f1_std=float(np.std(f1s)),
                         fit_seconds=round(float(np.mean(times)), 1),
                         n_seeds=len(seeds)))
        if acc_m > best[1]:
            best = (name, acc_m, pred0)

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    csv = os.path.join(RESULTS_DIR, "classification_metrics.csv")
    df.to_csv(csv, index=False)
    import json
    with open(csv.replace(".csv", "_params.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)  # provenance: params -> metrics
    print(f"\nSaved {csv}\n")
    print(df.to_string(index=False))

    # Accuracy bar chart with seed-variance error bars
    fig, ax = plt.subplots(figsize=(7, 4))
    yerr = df["accuracy_std"].to_numpy() if "accuracy_std" in df else None
    ax.bar(df["model"], df["accuracy"], yerr=yerr, color="#4C72B0",
           capsize=4, ecolor="#12263a")
    ax.set_ylabel("test accuracy (30-class)")
    ax.set_ylim(0, 1)
    ax.set_title(f"bacteria-ID classification: algorithm comparison "
                 f"(mean +/- std over {args.seeds} seeds)")
    for i, (v, e) in enumerate(zip(df["accuracy"], df.get("accuracy_std",
                                   [0] * len(df)), strict=False)):
        ax.text(i, v + e + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "classification_accuracy.png"), dpi=130)

    # Confusion matrix for the winner
    name, _, pred = best
    cm = confusion_matrix(y_te, pred, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"Confusion matrix (row-normalised) - {name}")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    labels = [STRAINS[i] for i in range(30)]
    ax.set_xticks(range(30)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(30)); ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "classification_confusion.png"), dpi=130)
    print(f"\nBest classifier: {name} (acc={best[1]:.4f})")


if __name__ == "__main__":
    main()
