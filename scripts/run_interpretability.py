"""Explainability (XAI): which wavenumbers drive the models?

Two complementary attributions, both mapped onto the wavenumber axis so they read
as peak importances a spectroscopist can interpret:

* SHAP (model-agnostic; TreeExplainer for the RandomForest) for both tasks,
* Integrated Gradients for the 1-D CNN (see raman_ml.interpretability), available
  via the library and exercised in the tests.

Outputs to benchmarks/plots/:
  shap_classification.png   per-wavenumber SHAP importance over a mean spectrum
  shap_quantification.png   same, for the concentration regressor

    python scripts/run_interpretability.py
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

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from raman_ml.datasets import load_bacteria_id, load_polystyrene
from raman_ml.interpretability import grad_cam_1d, shap_wavenumber_importance, top_peaks
from raman_ml.models import CNNClassifier
from raman_ml.preprocessing import l2_normalize, remove_baseline

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(_ROOT, "benchmarks", "plots")


def _plot(wn, importance, mean_spec, title, path, k=6):
    order = np.argsort(wn)
    wn, importance, mean_spec = wn[order], importance[order], mean_spec[order]
    imp_n = importance / (importance.max() + 1e-12)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(wn, mean_spec / (mean_spec.max() + 1e-12), color="0.6", lw=1,
            label="mean spectrum (scaled)")
    ax.fill_between(wn, imp_n, color="#C44E52", alpha=0.5,
                    label="SHAP importance (scaled)")
    peaks = top_peaks(importance, wn, k=k)
    for w, _ in peaks:
        ax.axvline(w, color="#C44E52", lw=0.6, ls=":")
        ax.annotate(f"{w:.0f}", (w, 1.02), fontsize=7, ha="center",
                    color="#C44E52")
    ax.set_xlabel("wavenumber (cm$^{-1}$)")
    ax.set_ylabel("scaled intensity / importance")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return peaks


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # --- classification: RandomForest on a bacteria-ID subsample ---
    Xc, yc, wn_c = load_bacteria_id("finetune")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xc), 1500, replace=False)
    rf_c = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                  random_state=0).fit(Xc[idx], yc[idx])
    imp_c = shap_wavenumber_importance(rf_c, Xc[idx], Xc[idx][:200])
    peaks_c = _plot(wn_c, imp_c, Xc[idx].mean(0),
                    "SHAP wavenumber importance - bacteria-ID (RandomForest)",
                    os.path.join(PLOTS_DIR, "shap_classification.png"))

    # --- quantification: RandomForest on the polystyrene set ---
    Xq_raw, _, conc, size_id, grid = load_polystyrene(1000)
    ylog = np.log10(conc); y = ylog.copy()
    for s in np.unique(size_id):
        y[size_id == s] = ylog[size_id == s] - ylog[size_id == s].max()
    Xq = l2_normalize(remove_baseline(Xq_raw, method="als"))
    rf_q = RandomForestRegressor(n_estimators=300, random_state=0).fit(Xq, y)
    imp_q = shap_wavenumber_importance(rf_q, Xq, Xq)
    peaks_q = _plot(grid, imp_q, Xq.mean(0),
                    "SHAP wavenumber importance - polystyrene conc. (RandomForest)",
                    os.path.join(PLOTS_DIR, "shap_quantification.png"))

    # --- Grad-CAM for a compact CNN on the same bacteria subsample ---
    try:
        cnn = CNNClassifier(n_out=30, epochs=12, batch_size=128, arch="resnet",
                            resnet_base=16, seed=0).fit(Xc[idx], yc[idx])
        cam = grad_cam_1d(cnn, Xc[idx][:200]).mean(0)
        _plot(wn_c, cam, Xc[idx].mean(0),
              "Grad-CAM wavenumber importance - bacteria-ID (1D-CNN)",
              os.path.join(PLOTS_DIR, "gradcam_classification.png"))
        print("Saved gradcam_classification.png")
    except Exception as e:  # noqa: BLE001
        print(f"Grad-CAM step skipped: {e}")

    print("Top SHAP wavenumbers (classification):",
          [f"{w:.0f}" for w, _ in peaks_c])
    print("Top SHAP wavenumbers (quantification):",
          [f"{w:.0f}" for w, _ in peaks_q])
    print("Saved shap_classification.png, shap_quantification.png to benchmarks/plots/")


if __name__ == "__main__":
    main()
