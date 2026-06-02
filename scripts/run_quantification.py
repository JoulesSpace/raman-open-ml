"""Quantification benchmark on the polystyrene dilution-series dataset.

Goal: predict log10(particle concentration) from a Raman spectrum, and compare
regression algorithms. The dataset is small (48 mean spectra over 8 particle
sizes x 6 concentrations), so we:

  * preprocess once (ALS baseline removal + L2 normalisation),
  * evaluate with repeated K-fold CV on the *real* mean spectra,
  * inside each training fold, enlarge the data with Gaussian replicates drawn
    from the per-point SD that ships with the dataset (real measurement spread),
  * always test on the real held-out spectra (no synthetic leakage).

    python scripts/run_quantification.py --n-aug 40 --cnn-epochs 40
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

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from raman_ml.datasets import load_polystyrene
from raman_ml.models import CNNRegressor
from raman_ml.preprocessing import remove_baseline
from raman_ml.variable_selection import VIPSelectedPLSR

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "benchmarks", "results")
PLOTS_DIR = os.path.join(_ROOT, "benchmarks", "plots")


class PLSWrapper:
    """PLSRegression with a 1-D predict, for a clean common interface."""

    def __init__(self, n_components):
        self.m = PLSRegression(n_components=n_components)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X).ravel()


def make_models(args):
    return {
        "PLSR": lambda: PLSWrapper(args.pls_components),
        "PLSR+VIP": lambda: VIPSelectedPLSR(args.pls_components, top_k=400),
        "PCR": lambda: make_pipeline(StandardScaler(),
                                     PCA(n_components=args.pls_components),
                                     LinearRegression()),
        "SVR-rbf": lambda: make_pipeline(StandardScaler(),
                                         SVR(C=10.0, gamma="scale")),
        "kNN": lambda: make_pipeline(StandardScaler(),
                                     KNeighborsRegressor(n_neighbors=5)),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=args.seed),
        "1D-CNN": lambda: CNNRegressor(epochs=args.cnn_epochs, batch_size=64,
                                       lr=1e-3, seed=args.seed),
    }


def preprocess(X_raw, SD_raw):
    """Baseline-remove + L2-normalise mean spectra; scale SD to the same domain.

    L2 normalisation is a per-spectrum linear scaling, so the SD (noise level)
    is divided by the same norm to stay consistent in the processed domain.
    """
    Xb = remove_baseline(X_raw, lam=1e5, p=0.01)
    norm = np.linalg.norm(Xb, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    Xp = Xb / norm
    SDp = SD_raw / norm
    return Xp, SDp


def augment(X, SD, y, n_aug, rng):
    Xs, ys = [X], [y]
    for _ in range(n_aug):
        Xs.append(X + rng.standard_normal(X.shape) * SD)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-aug", type=int, default=40,
                    help="replicate spectra generated per training sample")
    ap.add_argument("--pls-components", type=int, default=3)
    ap.add_argument("--cnn-epochs", type=int, default=40)
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--cv-repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("Loading polystyrene dilution series ...")
    X_raw, SD_raw, conc, size_id, grid = load_polystyrene(n_points=1000)
    # Target = relative concentration within each particle size's dilution
    # series (log10 of the dilution factor vs that series' most concentrated
    # sample). This is the single-analyte calibration framing: it removes the
    # inter-size confound (same particle count != same signal across sizes) and
    # mirrors a classic "predict the concentration of one analyte across a
    # dilution series" Raman task. See DATA_SOURCES.md.
    ylog = np.log10(conc)
    y = ylog.copy()
    for s in np.unique(size_id):
        m = size_id == s
        y[m] = ylog[m] - ylog[m].max()
    print(f"  {X_raw.shape[0]} spectra, {X_raw.shape[1]} points, "
          f"{len(np.unique(size_id))} particle sizes")
    print(f"  target = relative log10 conc, range {y.min():.2f} .. {y.max():.2f}")

    Xp, SDp = preprocess(X_raw, SD_raw)

    cv = RepeatedKFold(n_splits=args.cv_splits, n_repeats=args.cv_repeats,
                       random_state=args.seed)
    splits = list(cv.split(Xp))
    models = make_models(args)

    rows = []
    oof = {}  # name -> (per-sample averaged prediction) for parity plot
    for name, factory in models.items():
        print(f"\n>>> {name}")
        t0 = time.time()
        preds_sum = np.zeros(len(Xp))
        preds_cnt = np.zeros(len(Xp))
        fold_r2 = []
        all_true, all_pred = [], []
        for k, (tr, te) in enumerate(splits):
            rng = np.random.default_rng(args.seed * 1000 + k)
            Xa, ya = augment(Xp[tr], SDp[tr], y[tr], args.n_aug, rng)
            model = factory()
            model.fit(Xa, ya)
            p = model.predict(Xp[te])
            preds_sum[te] += p
            preds_cnt[te] += 1
            fold_r2.append(r2_score(y[te], p))
            all_true.extend(y[te]); all_pred.extend(p)
        dt = time.time() - t0
        all_true, all_pred = np.array(all_true), np.array(all_pred)
        r2 = r2_score(all_true, all_pred)
        rmse = np.sqrt(mean_squared_error(all_true, all_pred))
        mae = mean_absolute_error(all_true, all_pred)
        print(f"    R2(pooled)={r2:.3f}  RMSE={rmse:.3f} log-units  "
              f"MAE={mae:.3f}  fold-R2={np.mean(fold_r2):.3f}"
              f"+/-{np.std(fold_r2):.3f}  time={dt:.1f}s")
        rows.append(dict(model=name, R2=r2, RMSE_log10=rmse, MAE_log10=mae,
                         fold_R2_mean=np.mean(fold_r2),
                         fold_R2_std=np.std(fold_r2), time_s=round(dt, 1)))
        oof[name] = preds_sum / np.maximum(preds_cnt, 1)

    df = pd.DataFrame(rows).sort_values("R2", ascending=False)
    csv = os.path.join(RESULTS_DIR, "quantification_metrics.csv")
    df.to_csv(csv, index=False)
    import json
    with open(csv.replace(".csv", "_params.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)  # provenance: params -> metrics
    print(f"\nSaved {csv}\n")
    print(df.to_string(index=False))

    best = df.iloc[0]["model"]

    # R2 bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["model"], df["R2"].clip(lower=0), color="#55A868")
    ax.set_ylabel("R^2 (CV, relative log10 concentration)")
    ax.set_ylim(0, 1)
    ax.set_title("Polystyrene quantification: algorithm comparison")
    for i, v in enumerate(df["R2"]):
        ax.text(i, max(v, 0) + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "quantification_r2.png"), dpi=130)

    # Parity plot for the winner
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    sc = ax.scatter(y, oof[best], c=size_id, cmap="tab10", s=45,
                    edgecolor="k", linewidth=0.4)
    lims = [y.min() - 0.3, y.max() + 0.3]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("true relative log10 concentration")
    ax.set_ylabel("predicted relative log10 concentration")
    ax.set_title(f"Quantification parity - {best}")
    fig.colorbar(sc, label="particle-size id", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "quantification_parity.png"), dpi=130)
    print(f"\nBest regressor: {best} (R2={df.iloc[0]['R2']:.3f})")


if __name__ == "__main__":
    main()
