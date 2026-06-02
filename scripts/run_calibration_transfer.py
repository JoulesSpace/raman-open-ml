"""Calibration transfer for quantification, plus conformal prediction intervals.

Real Raman models trained on one instrument degrade on another (the dominant
field failure). The open polystyrene set is single-instrument, so we simulate a
*secondary instrument* with a documented, smooth response distortion (wavelength-
dependent gain + baseline drift + small wavenumber shift + noise) applied to the
same samples - giving paired transfer standards. We then show:

  1. a regressor trained on the primary instrument predicts well in-domain,
  2. it degrades on the (shifted) secondary instrument,
  3. Piecewise Direct Standardization (PDS), fit on a handful of paired
     standards, maps secondary spectra back and recovers accuracy.

We also wrap the in-domain model in split-conformal prediction intervals and
report empirical coverage. PDS + conformal are both absent from maintained OSS
Raman libraries.

    python scripts/run_calibration_transfer.py --n-standards 12
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from raman_ml import uncertainty as U
from raman_ml.calibration_transfer import (
    PiecewiseDirectStandardization,
    apply_transfer_if_improves,
    select_transfer_standards,
)
from raman_ml.datasets import load_polystyrene
from raman_ml.preprocessing import l2_normalize, remove_baseline

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "benchmarks", "results")
PLOTS_DIR = os.path.join(_ROOT, "benchmarks", "plots")


def secondary_instrument(X_raw, grid, seed=0):
    """Apply a documented instrument-response distortion that SURVIVES baseline +
    normalisation (so it is a genuine transfer problem, not one preprocessing
    silently fixes):

      * peak broadening (lower spectral resolution) via Gaussian convolution,
      * non-linear wavenumber-axis warp (miscalibration),
      * wavelength-dependent gain + baseline drift + mild noise.

    Returned spectra are PAIRED with the primary (same underlying samples).
    """
    from scipy.ndimage import gaussian_filter1d
    rng = np.random.default_rng(seed)
    L = X_raw.shape[1]
    t = np.linspace(0, 1, L)
    Y = gaussian_filter1d(X_raw, sigma=3.0, axis=1)          # resolution change
    base = np.arange(L, dtype=float)                          # axis warp
    warp = base + 6.0 * np.sin(np.pi * base / L)
    Y = np.vstack([np.interp(base, warp, row) for row in Y])
    gain = 1.0 + 0.4 * np.sin(2 * np.pi * t + 0.5)
    Y = Y * gain[None, :] + 0.15 * X_raw.max() * (t ** 2)[None, :]
    Y = Y + rng.normal(0, 0.01 * X_raw.std(), Y.shape)
    return Y


def prep(X_raw):
    return l2_normalize(remove_baseline(X_raw, method="als"))


def models():
    return {
        "PLSR": lambda: PLSWrap(3),
        "SVR-rbf": lambda: make_pipeline(StandardScaler(), SVR(C=10, gamma="scale")),
        "RandomForest": lambda: RandomForestRegressor(300, random_state=0),
    }


class PLSWrap:
    def __init__(self, n): self.m = PLSRegression(n_components=n)
    def fit(self, X, y): self.m.fit(X, y); return self
    def predict(self, X): return self.m.predict(X).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-standards", type=int, default=12)
    ap.add_argument("--half-window", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=25, help="random splits to average")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(PLOTS_DIR, exist_ok=True)

    X_raw, SD, conc, size_id, grid = load_polystyrene(1000)
    ylog = np.log10(conc); y = ylog.copy()
    for s in np.unique(size_id):
        y[size_id == s] = ylog[size_id == s] - ylog[size_id == s].max()

    Xp = prep(X_raw)                                   # primary instrument
    Xs = prep(secondary_instrument(X_raw, grid, args.seed))  # secondary instrument

    # n=48 is tiny, so a single split is noisy: average over many splits.
    idx = np.arange(len(Xp))
    acc = {name: {k: [] for k in ("in", "shift", "pds", "guard")}
           for name in models()}
    guard_pds_used = dict.fromkeys(models(), 0)
    for k in range(args.repeats):
        tr, te = train_test_split(idx, test_size=0.4, random_state=k,
                                  stratify=size_id)
        std = tr[select_transfer_standards(Xp[tr], n=args.n_standards, seed=k)]
        pds = PiecewiseDirectStandardization(half_window=args.half_window).fit(
            Xs[std], Xp[std])
        Xs_map = pds.transform(Xs)
        val = np.array([i for i in tr if i not in set(std.tolist())])  # guard val
        for name, factory in models().items():
            m = factory().fit(Xp[tr], y[tr])
            acc[name]["in"].append(r2_score(y[te], m.predict(Xp[te])))
            acc[name]["shift"].append(r2_score(y[te], m.predict(Xs[te])))
            acc[name]["pds"].append(r2_score(y[te], m.predict(Xs_map[te])))
            Xg, used = apply_transfer_if_improves(m, pds, Xs[val], y[val], Xs[te])
            acc[name]["guard"].append(r2_score(y[te], m.predict(Xg)))
            guard_pds_used[name] += int(used)

    rows = []
    print(f"(mean R2 over {args.repeats} splits)\n{'model':>14}  in-domain  "
          f"secondary(noTransfer)  secondary(PDS)  secondary(guarded)")
    for name in models():
        r2_in = float(np.mean(acc[name]["in"]))
        r2_shift = float(np.mean(acc[name]["shift"]))
        r2_pds = float(np.mean(acc[name]["pds"]))
        r2_guard = float(np.mean(acc[name]["guard"]))
        print(f"{name:>14}    {r2_in:5.3f}        {r2_shift:6.3f}            "
              f"{r2_pds:6.3f}         {r2_guard:6.3f} "
              f"(PDS used {guard_pds_used[name]}/{args.repeats})")
        rows.append(dict(model=name, r2_in_domain=r2_in,
                         r2_secondary_no_transfer=r2_shift,
                         r2_secondary_pds=r2_pds, r2_secondary_guarded=r2_guard,
                         pds_used_fraction=guard_pds_used[name] / args.repeats))

    # Jackknife+ prediction intervals for the in-domain model (fixed split)
    tr, te = train_test_split(idx, test_size=0.4, random_state=0, stratify=size_id)
    lo, hi = U.jackknife_plus_interval(
        lambda: RandomForestRegressor(200, random_state=0),
        Xp[tr], y[tr], Xp[te], alpha=args.alpha)
    im = U.interval_metrics(y[te], lo, hi)
    print(f"\nJackknife+ intervals (RF, alpha={args.alpha}): "
          f"coverage={im['coverage']:.3f} (target {1 - args.alpha:.2f}), "
          f"mean width={im['mean_width']:.3f} log-units")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "calibration_transfer_metrics.csv"),
              index=False)
    pd.DataFrame([dict(alpha=args.alpha, **im)]).to_csv(
        os.path.join(RESULTS_DIR, "conformal_interval_metrics.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(df)); w = 0.27
    ax.bar(x - w, df["r2_secondary_no_transfer"], w, label="secondary (no transfer)",
           color="#C44E52")
    ax.bar(x, df["r2_secondary_pds"], w, label="secondary (PDS, unguarded)",
           color="#8a94a6")
    ax.bar(x + w, df["r2_secondary_guarded"], w, label="secondary (guarded PDS)",
           color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(df["model"])
    ax.set_ylabel("R2 (relative log10 conc)")
    ax.set_ylim(min(-0.2, df["r2_secondary_pds"].min()), 1)
    ax.set_title("Calibration transfer: guarded PDS only transfers when it helps")
    ax.axhline(0, color="k", lw=0.6); ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "calibration_transfer.png"), dpi=130)
    print("Saved benchmarks/plots/calibration_transfer.png")


if __name__ == "__main__":
    main()
