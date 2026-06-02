"""Unsupervised PCA exploration of both datasets (chemometric EDA).

Standard chemometric PCA exploration: project spectra onto their principal
components, plot the score space
coloured by the label of interest, and show how much variance each component
captures. This is the standard first look at spectral structure before modelling.

Outputs to benchmarks/plots/:
  pca_scree.png            explained variance vs component, both datasets
  pca_bacteria.png         PC1-PC2 scores coloured by isolate (sampled classes)
  pca_polystyrene.png      PC1-PC2 scores coloured by concentration and by size

    python scripts/run_pca_explore.py
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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from raman_ml.datasets import load_bacteria_id, load_polystyrene
from raman_ml.preprocessing import l2_normalize, remove_baseline

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(_ROOT, "benchmarks", "plots")


def pca_scores(X, n=10):
    Z = StandardScaler().fit_transform(X)
    p = PCA(n_components=n, random_state=0)
    return p.fit_transform(Z), p.explained_variance_ratio_


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # --- bacteria-ID (sample a few isolates so the score plot is legible) ---
    Xb, yb, _ = load_bacteria_id("finetune")
    keep = np.unique(yb)[:6]
    m = np.isin(yb, keep)
    sb, evr_b = pca_scores(Xb[m], n=10)

    # --- polystyrene (preprocess from raw) ---
    Xp_raw, _, conc, size_id, _ = load_polystyrene(1000)
    Xp = l2_normalize(remove_baseline(Xp_raw, method="als"))
    sp, evr_p = pca_scores(Xp, n=10)

    # scree
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, 11), np.cumsum(evr_b) * 100, "o-", label="bacteria-ID")
    ax.plot(range(1, 11), np.cumsum(evr_p) * 100, "s-", label="polystyrene")
    ax.set_xlabel("number of principal components")
    ax.set_ylabel("cumulative variance explained (%)")
    ax.set_title("PCA scree: spectral variance is low-dimensional")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS_DIR, "pca_scree.png"), dpi=130)

    # bacteria score plot coloured by isolate
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(sb[:, 0], sb[:, 1], c=yb[m], cmap="tab10", s=12, alpha=0.7)
    ax.set_xlabel(f"PC1 ({evr_b[0] * 100:.0f}%)")
    ax.set_ylabel(f"PC2 ({evr_b[1] * 100:.0f}%)")
    ax.set_title("bacteria-ID PCA scores (6 isolates)")
    fig.colorbar(sc, label="isolate id", fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS_DIR, "pca_bacteria.png"), dpi=130)

    # polystyrene score plots: by concentration and by particle size
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    s0 = axes[0].scatter(sp[:, 0], sp[:, 1], c=np.log10(conc), cmap="viridis",
                         s=55, edgecolor="k", linewidth=0.4)
    axes[0].set_title("polystyrene PCA scores by concentration")
    fig.colorbar(s0, ax=axes[0], label="log10(particles/mL)")
    s1 = axes[1].scatter(sp[:, 0], sp[:, 1], c=size_id, cmap="tab10",
                         s=55, edgecolor="k", linewidth=0.4)
    axes[1].set_title("polystyrene PCA scores by particle size")
    fig.colorbar(s1, ax=axes[1], label="particle-size id")
    for ax in axes:
        ax.set_xlabel(f"PC1 ({evr_p[0] * 100:.0f}%)")
        ax.set_ylabel(f"PC2 ({evr_p[1] * 100:.0f}%)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "pca_polystyrene.png"), dpi=130)

    print("PCA explained variance (first 5 components):")
    print(f"  bacteria-ID : {np.round(evr_b[:5] * 100, 1)}  "
          f"(cum {np.cumsum(evr_b[:5])[-1] * 100:.0f}%)")
    print(f"  polystyrene : {np.round(evr_p[:5] * 100, 1)}  "
          f"(cum {np.cumsum(evr_p[:5])[-1] * 100:.0f}%)")
    print("Saved pca_scree.png, pca_bacteria.png, pca_polystyrene.png to benchmarks/plots/")


if __name__ == "__main__":
    main()
