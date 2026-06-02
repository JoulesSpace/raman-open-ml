"""Unified pipeline benchmark: combine every variation and find the best combo.

We tested baselines, normalisations, derivatives, dimensionality reduction and
models standalone. This brings them together: it sweeps the full grid

    baseline x normalisation x SG-derivative x dim-reduction x model

with leakage-safe cross-validation on the quantification task (relative log10
concentration), ranks every pipeline, and then runs Optuna HPO on the winning
model to squeeze out the last bit. The point is a single, honest "which
combination of our components actually works best" answer.

    python scripts/run_pipeline.py --tune
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from raman_ml.datasets import load_polystyrene
from raman_ml.preprocessing import l2_normalize, remove_baseline, savgol_derivative, snv
from raman_ml.tuning import tune

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "benchmarks", "results")
PLOTS_DIR = os.path.join(_ROOT, "benchmarks", "plots")

BASELINES = ["arpls", "als", "snip"]
NORMS = ["l2", "snv"]
SG = ["none", "d1"]
DR = ["none", "pca30"]
MODELS = ["PLSR", "SVR", "RF", "kNN"]


class PLSWrap(BaseEstimator, RegressorMixin):
    """sklearn-cloneable PLS with 1-D predict (for use inside a Pipeline + CV)."""

    def __init__(self, n=3):
        self.n = n

    def fit(self, X, y):
        self.m_ = PLSRegression(n_components=self.n).fit(X, y)
        return self

    def predict(self, X):
        return self.m_.predict(X).ravel()


def preprocess(X_raw, baseline, norm, sg):
    X = remove_baseline(X_raw, method=baseline)
    if sg == "d1":
        X = savgol_derivative(X, window=11, poly=2, deriv=1)
    return l2_normalize(X) if norm == "l2" else snv(X)


def build(dr, model):
    steps = []
    if dr.startswith("pca") or model in ("SVR", "kNN"):
        steps.append(StandardScaler())
    if dr.startswith("pca"):
        steps.append(PCA(n_components=int(dr[3:]), random_state=0))
    steps.append({"PLSR": PLSWrap(3), "SVR": SVR(C=10, gamma="scale"),
                  "RF": RandomForestRegressor(200, random_state=0),
                  "kNN": KNeighborsRegressor(5)}[model])
    return make_pipeline(*steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="Optuna-tune the winner")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(PLOTS_DIR, exist_ok=True)

    X_raw, _, conc, size_id, _ = load_polystyrene(1000)
    ylog = np.log10(conc); y = ylog.copy()
    for s in np.unique(size_id):
        y[size_id == s] = ylog[size_id == s] - ylog[size_id == s].max()
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=args.seed)

    # cache preprocessing per (baseline, norm, sg) - it is per-spectrum (no leakage)
    cache = {}
    rows = []
    for bl, nm, sg, dr, mdl in itertools.product(BASELINES, NORMS, SG, DR, MODELS):
        key = (bl, nm, sg)
        if key not in cache:
            cache[key] = preprocess(X_raw, bl, nm, sg)
        Xp = cache[key]
        try:
            r2 = cross_val_score(build(dr, mdl), Xp, y, cv=cv,
                                 scoring="r2").mean()
        except Exception:  # noqa: BLE001
            r2 = float("nan")
        rows.append(dict(baseline=bl, norm=nm, sg=sg, dr=dr, model=mdl, R2=r2))

    import pandas as pd
    df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(RESULTS_DIR, "pipeline_sweep.csv"), index=False)
    print(f"Swept {len(df)} pipelines. Top 12:\n")
    print(df.head(12).to_string(index=False))
    print("\nBest per model:")
    print(df.loc[df.groupby('model')['R2'].idxmax()].to_string(index=False))

    best = df.iloc[0]
    print(f"\nBEST PIPELINE: baseline={best.baseline} norm={best.norm} "
          f"sg={best.sg} dr={best.dr} model={best.model}  R2={best.R2:.3f}")

    if args.tune:
        Xb = preprocess(X_raw, best.baseline, best.norm, best.sg)
        spaces = {
            "RF": (RandomForestRegressor(random_state=0),
                   {"n_estimators": (200, 800), "max_depth": (3, 30),
                    "max_features": (0.1, 1.0), "min_samples_leaf": (1, 5)}),
            "SVR": (make_pipeline(StandardScaler(), SVR()),
                    {"svr__C": (1.0, 1000.0), "svr__gamma": (1e-4, 1.0),
                     "svr__epsilon": (1e-3, 0.2)}),
        }
        if best.model in spaces and best.dr == "none":
            est, space = spaces[best.model]
            _, params, score = tune(est, space, Xb, y, method="bayes",
                                    scoring="r2", cv=cv, n_iter=40)
            print(f"\nHPO on winner ({best.model}): R2 {best.R2:.3f} -> {score:.3f}")
            pp = {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in params.items()}
            print(f"  best params: {pp}")

    # plot top-12: encode the whole pipeline visually instead of a cryptic
    # "RF|als|l2|d1|none" label. colour = model, hatch = baseline, and three
    # on/off dots = the toggle steps (SNV, SG 1st-derivative, PCA).
    top = df.head(12).reset_index(drop=True)
    n = len(top)
    model_colors = {"RF": "#4C72B0", "SVR": "#55A868", "PLSR": "#C44E52", "kNN": "#8172B3"}
    baseline_hatch = {"als": "///", "arpls": "xxx", "snip": ".."}
    on_c, off_c = "#2a9d8f", "#cfd4dc"
    dot_cols = [("SNV", "norm", "snv"), ("SG d1", "sg", "d1"), ("PCA", "dr", "pca30")]
    dot_x = [1.05, 1.11, 1.17]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    for i, r in enumerate(top.itertuples()):
        y = n - 1 - i  # best at the top
        ax.barh(y, r.R2, color=model_colors.get(r.model, "#999999"),
                hatch=baseline_hatch.get(r.baseline, ""), edgecolor="white", linewidth=0.7)
        ax.text(r.R2 + 0.012, y, f"{r.R2:.3f}", va="center", ha="left",
                fontsize=7.5, color="black")
        for (_, col, on_val), xx in zip(dot_cols, dot_x, strict=True):
            engaged = getattr(r, col) == on_val
            ax.scatter(xx, y, s=80, color=on_c if engaged else off_c,
                       edgecolor="#444444", linewidth=0.4, zorder=5)
    for (label, _, _), xx in zip(dot_cols, dot_x, strict=True):
        ax.text(xx, n - 0.35, label, ha="center", va="bottom", fontsize=7.5)
    ax.axvline(1.0, color="0.75", lw=0.8, ls=":")
    ax.set_yticks(range(n)); ax.set_yticklabels(top["model"][::-1], fontsize=8)
    ax.set_xlim(0, 1.22); ax.set_ylim(-0.7, n + 0.1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("R$^2$ (CV)")
    ax.set_title("Unified pipeline sweep - top 12 by R$^2$\n"
                 "colour = model  |  hatch = baseline  |  dots SNV / SG-d1 / PCA "
                 "(green = on, grey = off)", fontsize=11)
    model_handles = [mpatches.Patch(facecolor=c, label=m) for m, c in model_colors.items()]
    base_handles = [mpatches.Patch(facecolor="white", edgecolor="#444", hatch=h, label=b)
                    for b, h in baseline_hatch.items()]
    leg1 = ax.legend(handles=model_handles, title="model", loc="upper left",
                     bbox_to_anchor=(1.01, 1.0), fontsize=8, title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=base_handles, title="baseline", loc="upper left",
              bbox_to_anchor=(1.01, 0.55), fontsize=8, title_fontsize=8)
    fig.savefig(os.path.join(PLOTS_DIR, "pipeline_sweep.png"), dpi=130, bbox_inches="tight")
    print("\nSaved benchmarks/results/pipeline_sweep.csv + plots/pipeline_sweep.png")


if __name__ == "__main__":
    main()
