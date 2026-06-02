"""Benchmark dimensionality-reduction methods for spectra (not just PCA).

PCA, t-SNE, UMAP and MDS are the standard exploratory embeddings. Here we run
them on equal footing and *quantify* how well each 2-D embedding preserves class structure
(silhouette + kNN cross-val accuracy in the embedding), so the choice is
evidence-based rather than eyeballed.

Outputs to benchmarks/:
  plots/dimreduction_bacteria.png       2-D embeddings coloured by isolate
  plots/dimreduction_polystyrene.png    2-D embeddings coloured by concentration
  results/dimreduction_separability.csv silhouette + kNN-acc per method/dataset

    python scripts/run_dimreduction.py
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

from raman_ml.datasets import load_bacteria_id, load_polystyrene
from raman_ml.dimensionality_reduction import embed_2d, embedding_separability
from raman_ml.preprocessing import l2_normalize, remove_baseline

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(_ROOT, "benchmarks", "plots")
RESULTS_DIR = os.path.join(_ROOT, "benchmarks", "results")


def _methods():
    methods = ["pca", "lda", "mds", "tsne"]
    try:
        import umap  # noqa: F401
        methods.append("umap")
    except ImportError:
        pass
    return methods


def _panel(X, y, color, title, path, rows):
    methods = _methods()
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4))
    for ax, mth in zip(axes, methods, strict=False):
        try:
            emb = embed_2d(X, method=mth, y=y, seed=0)
            sep = embedding_separability(emb, y)
            ax.scatter(emb[:, 0], emb[:, 1], c=color, cmap="tab10",
                       s=10, alpha=0.7)
            ax.set_title(f"{mth.upper()}\nsil={sep['silhouette']:.2f} "
                         f"kNN={sep['knn_accuracy']:.2f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            rows.append(dict(dataset=title, method=mth, **sep))
        except Exception as e:  # noqa: BLE001
            ax.set_title(f"{mth.upper()} failed", fontsize=9)
            print(f"  {title}/{mth} failed: {e}")
    fig.suptitle(f"Dimensionality reduction: {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    # bacteria: a few isolates for a legible embedding
    Xb, yb, _ = load_bacteria_id("finetune")
    keep = np.unique(yb)[:8]
    m = np.isin(yb, keep)
    rng = np.random.default_rng(0)
    idx = rng.choice(np.where(m)[0], min(1200, m.sum()), replace=False)
    _panel(Xb[idx], yb[idx], yb[idx], "bacteria-ID (8 isolates)",
           os.path.join(PLOTS_DIR, "dimreduction_bacteria.png"), rows)

    # polystyrene: colour by concentration, separability by particle size
    Xq_raw, _, conc, size_id, _ = load_polystyrene(1000)
    Xq = l2_normalize(remove_baseline(Xq_raw, method="als"))
    _panel(Xq, size_id, np.log10(conc), "polystyrene (by size)",
           os.path.join(PLOTS_DIR, "dimreduction_polystyrene.png"), rows)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "dimreduction_separability.csv"),
              index=False)
    print(df.to_string(index=False))
    print("\nHigher silhouette / kNN-accuracy = the embedding better preserves "
          "class structure. Saved plots + CSV to benchmarks/.")


if __name__ == "__main__":
    main()
