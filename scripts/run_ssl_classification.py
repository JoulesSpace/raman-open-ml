"""Self-supervised (masked-autoencoder) pretraining -> fine-tuned ensemble.

The research-flagged lever for the last accuracy gap. We pretrain a SpectralMAE on
UNLABELLED reference+finetune spectra (no labels, no test data), then fine-tune an
ensemble of classifier heads on the finetune labels and test with TTA. Reports the
honest result vs literature - SSL is claimed to exceed SOTA only if it genuinely
clears 0.878.

    python scripts/run_ssl_classification.py --members 5 --pretrain-epochs 40

See `src/raman_ml/ssl.py` for the SpectralMAE + MAEClassifier.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from sklearn.metrics import accuracy_score, f1_score

from raman_ml.augment import SpectralAugment
from raman_ml.datasets import load_bacteria_id
from raman_ml.ssl import MAEClassifier

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "benchmarks", "results")
LIT = {"Ho 2019": 0.822, "SANet 2026": 0.861, "open-world 2024": 0.878}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--pretrain-epochs", type=int, default=40)
    ap.add_argument("--finetune-epochs", type=int, default=40)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--encoder-path", default="", help="cache/reuse pretrained MAE")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    Xr, _, _ = load_bacteria_id("reference")
    Xf, yf, _ = load_bacteria_id("finetune")
    Xt, yt, _ = load_bacteria_id("test")
    X_unlab = np.vstack([Xr, Xf])              # inputs only, no labels, no test
    print(f"pretrain(unlabelled)={X_unlab.shape}  finetune={Xf.shape}  test={Xt.shape}")

    enc_path = args.encoder_path or os.path.join(
        RESULTS_DIR, f"mae_encoder_base{args.base}.pt")
    template = MAEClassifier(n_out=30, base=args.base,
                             pretrain_epochs=args.pretrain_epochs,
                             finetune_epochs=args.finetune_epochs, seed=args.seed)
    if os.path.exists(enc_path):
        template.load_encoder(enc_path)
        print(f"  loaded cached MAE encoder: {enc_path}")
    else:
        t0 = time.time()
        template.pretrain(X_unlab)
        template.save_encoder(enc_path)
        print(f"  MAE pretraining done ({time.time() - t0:.0f}s), cached -> {enc_path}")

    tta = SpectralAugment(noise=0.005, offset=0.005, slope=0.005,
                          mult=(0.98, 1.02), shift=2, p=0.7, seed=99)

    def predict_tta(clf, X):
        ps = [clf.predict_proba(X)]
        for _ in range(args.tta):
            ps.append(clf.predict_proba(tta(X)))
        return np.mean(ps, axis=0)

    probs, accs = [], []
    for m in range(args.members):
        member = MAEClassifier(n_out=30, base=args.base,
                               finetune_epochs=args.finetune_epochs,
                               seed=args.seed + m)
        member.mae = copy.deepcopy(template.mae)      # shared SSL init
        member.n_in_ = template.n_in_
        member.fit(Xf, yf)
        pt = predict_tta(member, Xt)
        probs.append(pt)
        accs.append(accuracy_score(yt, pt.argmax(1)))
        print(f"  member {m + 1}/{args.members}: test acc={accs[-1]:.4f}")

    ens = np.mean(probs, axis=0)
    acc = accuracy_score(yt, ens.argmax(1))
    f1 = f1_score(yt, ens.argmax(1), average="macro")
    print(f"\nsingle SSL member mean = {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"SSL ENSEMBLE ({args.members}) + TTA = {acc:.4f}  macroF1 = {f1:.4f}")
    for k, v in LIT.items():
        print(f"  vs {k:<16} {v:.3f}{'  <-- we beat this' if acc > v else ''}")

    import pandas as pd
    pd.DataFrame([dict(members=args.members, single_mean=float(np.mean(accs)),
                       single_std=float(np.std(accs)), ssl_ensemble_acc=acc,
                       ssl_ensemble_f1=f1)]).to_csv(
        os.path.join(RESULTS_DIR, "ssl_classification_metrics.csv"), index=False)
    import json
    with open(os.path.join(RESULTS_DIR, "ssl_classification_params.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)
    print("\nSaved benchmarks/results/ssl_classification_metrics.csv")


if __name__ == "__main__":
    main()
