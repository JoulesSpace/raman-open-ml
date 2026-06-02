"""Generative augmentation benchmark: which synthesizer helps a few-shot model?

Generative augmentation only earns its keep when labelled data is scarce, so we
build an honest few-shot setup on bacteria-ID (keep ``--shots`` spectra per
isolate, 30 classes) and ask whether adding synthetic spectra lifts a downstream
classifier evaluated on the full held-out test split. Every generator only sees
the few-shot real data, then we train LogisticRegression on real + synthetic.

Two families are compared:

  * general tabular synthesizers via the **tabgan** library, applied in PCA space
    (the tractable way to use them on ~1000 correlated channels):
        random / OriginalGenerator   - resampling baseline
        Bayesian (Gaussian copula)   - BayesianGenerator
        CTGAN                        - tabular GAN
        forest diffusion             - ForestDiffusionGenerator
  * purpose-built **1-D convolutional** models at full spectral resolution:
        WGAN-GP    - raman_ml.generative.SpectralGAN
        diffusion  - raman_ml.generative.SpectralDiffusion (DDPM)

plus two references: no augmentation, and classical domain augmentation
(raman_ml.augment.SpectralAugment). The hypothesis the plot tests: a conv model
that respects wavenumber locality beats both generic tabular synthesizers and an
adversarial GAN on this small, smooth-signal problem.

    python scripts/run_generative_augmentation.py --shots 20 --gen 40 --epochs 300

Outputs:
  results/generative_augmentation_metrics.csv
  plots/generated_spectra.png            real vs generated mean spectra (3 classes)
  plots/generative_augmentation.png      test accuracy by augmentation method
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from raman_ml.augment import SpectralAugment  # noqa: E402
from raman_ml.datasets import load_bacteria_id  # noqa: E402
from raman_ml.generative import SpectralDiffusion, SpectralGAN  # noqa: E402
from raman_ml.preprocessing import l2_normalize  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "benchmarks", "results")
PLOTS_DIR = os.path.join(_ROOT, "benchmarks", "plots")
CONV = ("WGAN-GP (1D conv)", "diffusion (1D conv)",
        "classical + WGAN-GP", "classical + diffusion")  # our conv models, for highlight


def _few_shot(X, y, shots, seed):
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(y == c)[0],
                                     size=min(shots, int((y == c).sum())), replace=False)
                          for c in np.unique(y)])
    rng.shuffle(idx)
    return X[idx], y[idx]


def _fit_eval(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=3000, C=5.0).fit(Xtr, ytr)
    p = clf.predict(Xte)
    return accuracy_score(yte, p), f1_score(yte, p, average="macro")


def _classical_aug(X, y, per_class, seed):
    aug = SpectralAugment(noise=0.01, offset=0.01, slope=0.01, shift=3, warp=2.0,
                          p=0.7, seed=seed)
    reps = max(1, per_class // max(1, len(X) // len(np.unique(y))))
    return np.concatenate([aug(X) for _ in range(reps)], axis=0), np.tile(y, reps)


def _tabgan_synth(gen, Z, ytr, Zte, n_orig, pca):
    """Run a tabgan generator in PCA space; return inverse-PCA synthetic (X, y)."""
    import pandas as pd
    cols = [f"p{i}" for i in range(Z.shape[1])]
    nt, ntgt = gen.generate_data_pipe(pd.DataFrame(Z, columns=cols),
                                      pd.DataFrame({"t": ytr}),
                                      pd.DataFrame(Zte, columns=cols))
    nt = np.asarray(nt, dtype=np.float32)
    ntgt = np.asarray(ntgt).ravel()
    syn_Z, syn_y = nt[n_orig:], ntgt[n_orig:]  # rows appended beyond the originals
    if len(syn_Z) == 0:
        return None
    return pca.inverse_transform(syn_Z).astype(np.float32), syn_y.astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument("--gen", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--pca", type=int, default=30, help="PCA dims for tabular methods")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-tabgan", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    Xf, yf, wn = load_bacteria_id("finetune")
    Xt, yt, _ = load_bacteria_id("test")
    n_classes = int(yf.max()) + 1
    Xfs, yfs = _few_shot(Xf, yf, args.shots, args.seed)
    Xfs_n, Xt_n = l2_normalize(Xfs), l2_normalize(Xt)
    print(f"Few-shot: {len(Xfs)} spectra ({args.shots}/class), test={len(Xt)}\n")

    results, diff_syn = [], None

    def add(name, Xe, ye):
        if Xe is None or len(Xe) == 0:
            acc, f1 = _fit_eval(Xfs_n, yfs, Xt_n, yt); n_tr = len(Xfs_n)
        else:
            acc, f1 = _fit_eval(np.vstack([Xfs_n, Xe]), np.concatenate([yfs, ye]),
                                Xt_n, yt); n_tr = len(Xfs_n) + len(Xe)
        results.append(dict(method=name, n_train=n_tr, accuracy=acc, macro_f1=f1))
        print(f"  {name:24s} acc={acc:.3f}  macroF1={f1:.3f}  (n_train={n_tr})")

    add("real only", None, None)
    Xca, yca = _classical_aug(Xfs_n, yfs, args.gen, args.seed)
    add("classical aug", Xca, yca)

    # --- tabgan family, in PCA space ---
    if not args.no_tabgan:
        pca = PCA(n_components=min(args.pca, len(Xfs_n) - 1), random_state=args.seed)
        Z = pca.fit_transform(Xfs_n)
        Zte = pca.transform(Xt_n)
        from tabgan.sampler import (
            BayesianGenerator,
            ForestDiffusionGenerator,
            GANGenerator,
            OriginalGenerator,
        )
        tg = [("random (tabgan)", OriginalGenerator),
              ("Bayesian copula", BayesianGenerator),
              ("CTGAN (tabular)", GANGenerator),
              ("forest diffusion", ForestDiffusionGenerator)]
        for name, G in tg:
            t0 = time.time()
            try:
                syn = _tabgan_synth(G(), Z, yfs, Zte, len(Z), pca)
                add(name, *(syn if syn else (None, None)))
                print(f"    [{name} in {time.time() - t0:.0f}s]")
            except Exception as e:  # noqa: BLE001
                print(f"  {name:24s} skipped: {repr(e)[:120]}")

    # --- purpose-built 1-D conv models, full resolution ---
    t0 = time.time()
    gan = SpectralGAN(n_classes, epochs=args.epochs, seed=args.seed, verbose=True).fit(Xfs_n, yfs)
    Xgan, ygan = gan.generate_balanced(args.gen)
    add("WGAN-GP (1D conv)", Xgan, ygan)
    print(f"    [WGAN-GP in {time.time() - t0:.0f}s]")

    t0 = time.time()
    diff = SpectralDiffusion(n_classes=n_classes, timesteps=args.timesteps,
                             epochs=args.epochs, seed=args.seed, verbose=True).fit(Xfs_n, yfs)
    Xdf, ydf = diff.generate_balanced(args.gen)
    diff_syn = (Xdf, ydf)
    add("diffusion (1D conv)", Xdf, ydf)
    print(f"    [diffusion in {time.time() - t0:.0f}s]")

    # --- stacked: classical augmentation + a generative model on top ---
    add("classical + WGAN-GP", np.vstack([Xca, Xgan]), np.concatenate([yca, ygan]))
    add("classical + diffusion", np.vstack([Xca, Xdf]), np.concatenate([yca, ydf]))

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "generative_augmentation_metrics.csv"), index=False)
    best = df.loc[df["accuracy"].idxmax()]
    print(f"\nBest augmentation: {best['method']} (acc={best['accuracy']:.3f})")

    # --- plot: real vs diffusion-generated mean spectra for 3 classes ---
    Xdf, ydf = diff_syn
    classes = [int(c) for c in np.unique(yfs)[:3]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, c in zip(axes, classes, strict=True):
        ax.plot(wn, Xfs_n[yfs == c].mean(0), color="#12263a", lw=1.1, label="real mean")
        ax.plot(wn, Xdf[ydf == c].mean(0), color="#2a9d8f", lw=1.1, ls="--",
                label="diffusion mean")
        ax.set_title(f"isolate {c}", fontsize=9)
        ax.set_xlabel("wavenumber (cm$^{-1}$)"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("L2-normalised intensity")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Real vs diffusion-generated mean spectra (few-shot bacteria-ID)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(PLOTS_DIR, "generated_spectra.png"), dpi=130)
    plt.close(fig)

    # --- plot: accuracy by method ---
    ds = df.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    colors = ["#2a9d8f" if m in CONV else "#9aa3b2" for m in ds["method"]]
    ax.barh(range(len(ds)), ds["accuracy"], color=colors, edgecolor="#12263a", linewidth=0.5)
    for i, v in enumerate(ds["accuracy"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(range(len(ds))); ax.set_yticklabels(ds["method"], fontsize=8.5)
    ax.set_xlabel("test accuracy"); ax.set_xlim(0, max(df["accuracy"]) * 1.16)
    ax.set_title(f"Few-shot bacteria-ID ({args.shots}/class): generative augmentation\n"
                 "teal = purpose-built 1-D conv models; grey = tabular / baseline", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "generative_augmentation.png"), dpi=130)
    plt.close(fig)
    print("\nSaved generative_augmentation_metrics.csv + 2 plots")


if __name__ == "__main__":
    main()
