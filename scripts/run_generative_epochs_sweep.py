"""Training-budget sensitivity: is the GAN-vs-diffusion result an epoch artefact?

A fair generator comparison must control training budget, so this sweeps the
number of training epochs for the two purpose-built 1-D conv generators (and
CTGAN via tabgan) on the same few-shot bacteria-ID setup, and plots downstream
test accuracy against epochs. If the DDPM simply needed longer, its curve should
keep climbing; if it has converged, the curve flattens (and its training loss
stops dropping).

    python scripts/run_generative_epochs_sweep.py --budgets 150 300 700 1500

Outputs:
  results/generative_epochs_sweep.csv
  plots/generative_epochs_sweep.png
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

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

from raman_ml.datasets import load_bacteria_id  # noqa: E402
from raman_ml.generative import SpectralDiffusion, SpectralGAN  # noqa: E402
from raman_ml.preprocessing import l2_normalize  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "benchmarks", "results")
PLOTS_DIR = os.path.join(_ROOT, "benchmarks", "plots")


def _few_shot(X, y, shots, seed):
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(y == c)[0],
                                     size=min(shots, int((y == c).sum())), replace=False)
                          for c in np.unique(y)])
    rng.shuffle(idx)
    return X[idx], y[idx]


def _acc(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=3000, C=5.0).fit(Xtr, ytr)
    return accuracy_score(yte, clf.predict(Xte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument("--gen", type=int, default=40)
    ap.add_argument("--budgets", type=int, nargs="+", default=[150, 300, 700, 1500])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    Xf, yf, _ = load_bacteria_id("finetune")
    Xt, yt, _ = load_bacteria_id("test")
    n_classes = int(yf.max()) + 1
    Xfs, yfs = _few_shot(Xf, yf, args.shots, args.seed)
    Xfs_n, Xt_n = l2_normalize(Xfs), l2_normalize(Xt)
    real_acc = _acc(Xfs_n, yfs, Xt_n, yt)
    print(f"real-only acc={real_acc:.3f}\n")

    rows = []
    for b in args.budgets:
        t0 = time.time()
        gan = SpectralGAN(n_classes, epochs=b, seed=args.seed).fit(Xfs_n, yfs)
        Xg, yg = gan.generate_balanced(args.gen)
        a_gan = _acc(np.vstack([Xfs_n, Xg]), np.concatenate([yfs, yg]), Xt_n, yt)

        diff = SpectralDiffusion(n_classes=n_classes, epochs=b, seed=args.seed).fit(Xfs_n, yfs)
        Xd, yd = diff.generate_balanced(args.gen)
        a_dif = _acc(np.vstack([Xfs_n, Xd]), np.concatenate([yfs, yd]), Xt_n, yt)

        rows.append(dict(epochs=b, gan_acc=a_gan, diffusion_acc=a_dif,
                         diffusion_loss=round(float(diff.final_loss_), 4),
                         gan_critic_loss=round(float(gan.final_critic_loss_), 3)))
        print(f"  epochs={b:5d}  GAN={a_gan:.3f}  diffusion={a_dif:.3f}  "
              f"(diff_loss={diff.final_loss_:.3f})  [{time.time() - t0:.0f}s]")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "generative_epochs_sweep.csv"), index=False)
    print("\n" + df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(df["epochs"], df["gan_acc"], "-o", color="#2a9d8f", label="WGAN-GP (1D conv)")
    ax.plot(df["epochs"], df["diffusion_acc"], "-s", color="#9b5de5", label="diffusion (DDPM)")
    ax.axhline(real_acc, color="#9aa3b2", ls="--", lw=1, label=f"real only ({real_acc:.3f})")
    ax.axhline(0.694, color="#12263a", ls=":", lw=1, label="classical aug (0.694)")
    ax.set_xlabel("training epochs"); ax.set_ylabel("test accuracy")
    ax.set_title("Generative augmentation vs training budget (few-shot bacteria-ID)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS_DIR, "generative_epochs_sweep.png"), dpi=130)
    print("\nSaved generative_epochs_sweep.csv + generative_epochs_sweep.png")


if __name__ == "__main__":
    main()
