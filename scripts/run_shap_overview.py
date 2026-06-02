"""Per-class SHAP overview: which Raman bands distinguish each of the 30 isolates.

Trains a RandomForest on the (in-distribution) finetune split, computes per-class
SHAP attributions, and renders a (30 isolates x wavenumber) heatmap so you can see
at a glance which spectral regions drive each substance's identification, plus a
table of each isolate's top discriminative bands.

    python scripts/run_shap_overview.py
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from sklearn.ensemble import RandomForestClassifier

from raman_ml.datasets import STRAINS, load_bacteria_id
from raman_ml.interpretability import shap_per_class_importance

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(_ROOT, "benchmarks", "plots")
RESULTS_DIR = os.path.join(_ROOT, "benchmarks", "results")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X, y, wn = load_bacteria_id("finetune")        # in-distribution, all 30 classes
    rng = np.random.default_rng(0)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(X, y)

    # explain a balanced subset (~25 spectra/class) for per-class attributions
    expl_idx = np.concatenate([
        rng.choice(np.where(y == c)[0], size=min(25, np.sum(y == c)),
                   replace=False) for c in range(30)])
    print(f"computing per-class SHAP on {len(expl_idx)} spectra ...")
    imp = shap_per_class_importance(rf, X, X[expl_idx], y[expl_idx])  # (30, L)

    order = np.argsort(wn)                          # ascending wavenumber for display
    wn_s, imp_s = wn[order], imp[:, order]
    row_norm = imp_s / (imp_s.max(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(row_norm, aspect="auto", cmap="magma", origin="lower",
                   extent=[wn_s.min(), wn_s.max(), -0.5, 29.5])
    ax.set_yticks(range(30))
    ax.set_yticklabels([STRAINS[i] for i in range(30)], fontsize=7)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_title("Per-class SHAP wavenumber importance\n"
                 "(which bands distinguish each of the 30 isolates; row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                 label="relative SHAP importance")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "shap_classification_byclass.png"), dpi=140)

    # top-3 discriminative bands per isolate -> CSV
    import pandas as pd
    rows = []
    for c in range(30):
        top = np.argsort(-imp[c])[:3]
        rows.append(dict(isolate=STRAINS[c],
                         top_bands_cm1=", ".join(f"{wn[i]:.0f}" for i in top)))
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, "shap_per_class_bands.csv"), index=False)
    print("Saved benchmarks/plots/shap_classification_byclass.png and "
          "benchmarks/results/shap_per_class_bands.csv")
    for r in rows[:6]:
        print(f"  {r['isolate']:<22} {r['top_bands_cm1']}")


if __name__ == "__main__":
    main()
