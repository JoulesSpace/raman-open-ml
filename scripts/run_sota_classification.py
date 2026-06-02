"""SOTA-protocol bacteria-ID classification: pretrain -> fine-tune -> ensemble.

The published bacteria-ID numbers (Ho et al. 2019: 82.2%; SANet 86.1%; open-world
SE-ResNet ensemble 87.8%) all use the proper protocol: pretrain on the 60k
`reference` set, fine-tune on `finetune`, evaluate on the held-out `test` set.
Our flat comparison skipped the pretraining step; this script does it right with
an SE-ResNet deep ensemble + augmentation, and adds the trust layer on top.

    python scripts/run_sota_classification.py --members 5 --pretrain-epochs 30 \
        --finetune-epochs 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from sklearn.metrics import accuracy_score, f1_score

from raman_ml import uncertainty as U
from raman_ml.augment import SpectralAugment
from raman_ml.datasets import load_bacteria_id
from raman_ml.models import CNNClassifier

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "benchmarks", "results")

LITERATURE = {"Ho 2019 ResNet": 0.822, "SANet 2026": 0.861,
              "SE-ResNet ensemble 2024": 0.878}
PLOTS_DIR = os.path.join(_ROOT, "benchmarks", "plots")


def plot_leaderboard(our_acc, path):
    """Quality-only leaderboard: our accuracy vs published numbers.

    Deliberately NOT a cost-vs-quality plot: the papers do not report comparable
    training cost (different hardware / data sizes), so a cost axis would be
    fabricated. Accuracy is the honest common ground.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bars = [("Ho 2019\nResNet", 0.822, False),
            ("SANet\n2026", 0.861, False),
            ("this repo\nens. + TTA", our_acc, True),
            ("SE-ResNet ens.\n2024 (open-world)", 0.878, False)]
    bars.sort(key=lambda b: b[1])
    labels = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    colors = ["#2a9d8f" if b[2] else "#9aa3b2" for b in bars]
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    ax.bar(range(len(bars)), vals, color=colors, edgecolor="#12263a", linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.0015, f"{v:.3f}", ha="center", fontsize=9,
                fontweight="bold" if bars[i][2] else "normal")
    ax.set_xticks(range(len(bars))); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0.80, 0.895); ax.set_ylabel("test accuracy")
    ax.set_title("Bacteria-ID classification vs literature "
                 "(pretrain -> fine-tune -> test)\n"
                 "accuracy only; training cost not comparable across papers",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--pretrain-epochs", type=int, default=30)
    ap.add_argument("--finetune-epochs", type=int, default=30)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--arch", choices=["resnet", "msresnet", "both"],
                    default="both")
    ap.add_argument("--tta", type=int, default=0,
                    help="test-time augmentation passes (0 = off)")
    ap.add_argument("--pretrain-size", type=int, default=60000)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    Xr, yr, _ = load_bacteria_id("reference", subsample=args.pretrain_size,
                                 seed=args.seed)
    Xf, yf, _ = load_bacteria_id("finetune")
    Xt, yt, _ = load_bacteria_id("test")
    # small calibration slice from finetune for the trust layer (not used to train)
    rng = np.random.default_rng(args.seed)
    cal_idx = rng.choice(len(Xf), size=600, replace=False)
    cal_mask = np.zeros(len(Xf), bool); cal_mask[cal_idx] = True
    Xf_tr, yf_tr = Xf[~cal_mask], yf[~cal_mask]
    Xf_cal, yf_cal = Xf[cal_mask], yf[cal_mask]
    print(f"pretrain={Xr.shape}  finetune={Xf_tr.shape}  cal={Xf_cal.shape}  "
          f"test={Xt.shape}\n")

    # mild test-time augmentation: average probs over augmented copies
    tta_aug = SpectralAugment(noise=0.005, offset=0.005, slope=0.005,
                              mult=(0.98, 1.02), shift=2, p=0.7, seed=99)

    def predict_tta(clf, X):
        ps = [clf.predict_proba(X)]
        for _ in range(args.tta):
            ps.append(clf.predict_proba(tta_aug(X)))
        return np.mean(ps, axis=0)

    probs_test, probs_cal = [], []
    member_acc = []
    for m in range(args.members):
        t0 = time.time()
        # heterogeneous ensemble: alternate plain SE-ResNet and multi-scale
        arch = args.arch
        if arch == "both":
            arch = "resnet" if m % 2 == 0 else "msresnet"
        layers = (2, 2, 2) if arch == "msresnet" else (2, 2, 2, 2)
        aug = SpectralAugment(noise=0.01, offset=0.01, slope=0.01,
                              mult=(0.97, 1.03), shift=3, p=0.5, seed=args.seed + m)
        clf = CNNClassifier(n_out=30, epochs=args.pretrain_epochs, batch_size=256,
                            arch=arch, resnet_base=args.base,
                            resnet_layers=layers, se=True, label_smoothing=0.05,
                            augment=aug, seed=args.seed + m)
        clf.fit(Xr, yr)                                   # pretrain on reference
        clf.finetune(Xf_tr, yf_tr, epochs=args.finetune_epochs, lr=1e-4)  # adapt
        pt = predict_tta(clf, Xt)
        probs_test.append(pt)
        probs_cal.append(predict_tta(clf, Xf_cal))
        acc = accuracy_score(yt, pt.argmax(1))
        member_acc.append(acc)
        print(f"  member {m + 1}/{args.members} [{arch}]: test acc={acc:.4f}  "
              f"({time.time() - t0:.0f}s)")

    ens_test = np.mean(probs_test, axis=0)
    ens_cal = np.mean(probs_cal, axis=0)
    acc_ens = accuracy_score(yt, ens_test.argmax(1))
    f1_ens = f1_score(yt, ens_test.argmax(1), average="macro")
    print(f"\nsingle-model mean acc = {np.mean(member_acc):.4f} "
          f"+/- {np.std(member_acc):.4f}")
    print(f"ENSEMBLE ({args.members}) test acc = {acc_ens:.4f}  macroF1 = {f1_ens:.4f}")
    print("\n  vs literature (official bacteria-ID 30-class test):")
    for k, v in LITERATURE.items():
        mark = "  <-- we beat this" if acc_ens > v else ""
        print(f"    {k:<26} {v:.3f}{mark}")

    # trust layer on the ensemble probabilities (calibrate on finetune cal slice)
    eps = 1e-8
    cal_logits = np.log(ens_cal + eps)
    T = U.fit_temperature(cal_logits, yf_cal)
    ece_before = U.expected_calibration_error(ens_test, yt)
    ece_after = U.expected_calibration_error(
        U.softmax(np.log(ens_test + eps), T=T), yt)
    fn, _ = U.calibrate_conformal_classifier(
        U.softmax(cal_logits, T=T), yf_cal, alpha=args.alpha, raps=True,
        seed=args.seed)
    masks = fn(U.softmax(np.log(ens_test + eps), T=T))
    sm = U.set_metrics(masks, yt)
    print(f"\n  trust layer: ECE {ece_before:.3f}->{ece_after:.3f}, "
          f"conformal coverage={sm['coverage']:.3f} (target {1 - args.alpha:.2f}), "
          f"set size={sm['avg_set_size']:.2f}/30")

    import pandas as pd
    pd.DataFrame([dict(members=args.members,
                       single_mean_acc=float(np.mean(member_acc)),
                       single_std=float(np.std(member_acc)),
                       ensemble_acc=acc_ens, ensemble_macro_f1=f1_ens,
                       ece_before=ece_before, ece_after=ece_after,
                       conformal_coverage=sm["coverage"],
                       conformal_set_size=sm["avg_set_size"],
                       **{f"lit_{k.split()[0]}": v for k, v in LITERATURE.items()})
                  ]).to_csv(os.path.join(RESULTS_DIR,
                                         "sota_classification_metrics.csv"),
                            index=False)
    import json
    with open(os.path.join(RESULTS_DIR, "sota_classification_params.json"),
              "w") as fh:
        json.dump(vars(args), fh, indent=2)  # provenance: params -> metrics
    plot_leaderboard(acc_ens, os.path.join(PLOTS_DIR, "sota_leaderboard.png"))
    print("\nSaved benchmarks/results/sota_classification_metrics.csv "
          "+ plots/sota_leaderboard.png")


if __name__ == "__main__":
    main()
