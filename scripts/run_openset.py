"""Open-set recognition: reject unknown isolates the model was never trained on.

The #1 stated clinical concern for Raman classifiers (Lebron et al. 2024) is that
a closed-set softmax confidently mislabels an unseen species as a known one. We
hold out a subset of the 30 isolates as "unknown", train a 1D-ResNet on the rest
(in-distribution finetune campaign to avoid confounding with domain shift), and
measure how well three post-hoc detectors flag the unknowns.

Reports closed-set accuracy on knowns plus AUROC and FPR@95%TPR for MSP, energy,
and Mahalanobis detectors. No maintained OSS Raman library ships this.

    python scripts/run_openset.py --n-unknown 6 --cnn-epochs 40
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from sklearn.metrics import accuracy_score

from raman_ml import ood as O
from raman_ml import uncertainty as U
from raman_ml.augment import SpectralAugment
from raman_ml.datasets import load_bacteria_id
from raman_ml.models import CNNClassifier

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "benchmarks", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-unknown", type=int, default=6)
    ap.add_argument("--cnn-epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # In-distribution campaign: train on finetune, evaluate on test.
    Xtr, ytr, _ = load_bacteria_id("finetune")
    Xte, yte, _ = load_bacteria_id("test")

    rng = np.random.default_rng(args.seed)
    classes = np.arange(30)
    unknown = np.sort(rng.choice(classes, size=args.n_unknown, replace=False))
    known = np.array([c for c in classes if c not in unknown])
    remap = {c: i for i, c in enumerate(known)}
    print(f"unknown (held-out) isolates: {list(unknown)}")
    print(f"known: {len(known)} classes, training a 1D-ResNet ...")

    tr_mask = np.isin(ytr, known)
    Xk, yk = Xtr[tr_mask], np.array([remap[c] for c in ytr[tr_mask]])
    aug = SpectralAugment(noise=0.01, offset=0.01, slope=0.01,
                          mult=(0.97, 1.03), shift=3, p=0.5, seed=args.seed)
    clf = CNNClassifier(n_out=len(known), epochs=args.cnn_epochs, batch_size=256,
                        arch="resnet", resnet_base=32, label_smoothing=0.05,
                        augment=aug, seed=args.seed)
    clf.fit(Xk, yk)

    # Test split: known vs unknown
    te_known = np.isin(yte, known)
    Xk_te, yk_te = Xte[te_known], np.array([remap[c] for c in yte[te_known]])
    Xu_te = Xte[~te_known]

    acc = accuracy_score(yk_te, clf.predict(Xk_te))
    print(f"\nclosed-set accuracy on known test isolates: {acc:.3f}")

    # OOD scores
    log_k, log_u = clf.logits(Xk_te), clf.logits(Xu_te)
    p_k, p_u = U.softmax(log_k), U.softmax(log_u)
    maha = O.MahalanobisOOD().fit(clf.embed(Xk), yk)
    detectors = {
        "MSP": (O.msp_score(p_k), O.msp_score(p_u)),
        "Energy": (O.energy_score(log_k), O.energy_score(log_u)),
        "Mahalanobis": (maha.score(clf.embed(Xk_te)), maha.score(clf.embed(Xu_te))),
    }

    import pandas as pd
    rows = []
    print("\n  detector      AUROC   FPR@95TPR")
    for name, (s_in, s_ood) in detectors.items():
        au = O.auroc(s_in, s_ood)
        fpr = O.fpr_at_tpr(s_in, s_ood, 0.95)
        print(f"  {name:<12} {au:.3f}   {fpr:.3f}")
        rows.append(dict(detector=name, auroc=au, fpr_at_95tpr=fpr))
    df = pd.DataFrame(rows)
    df.insert(0, "closed_set_acc", acc)
    df.insert(0, "n_unknown", args.n_unknown)
    df.to_csv(os.path.join(RESULTS_DIR, "openset_metrics.csv"), index=False)
    print("\nSaved benchmarks/results/openset_metrics.csv")
    print("AUROC ~ 1.0 = perfect unknown rejection; FPR@95 = known flagged as "
          "unknown when catching 95% of unknowns.")


if __name__ == "__main__":
    main()
