"""Domain-shift-aware classification + trust layer (the repo's headline study).

bacteria-ID's `reference` set and its `finetune`/`test` sets are different
measurement campaigns. A model trained on `reference` scores ~90% in-distribution
but collapses on `test`. This is the field's #1 open problem (instrument/campaign
shift); see agent-memory/insights/bacteria-id-domain-shift.md.

This script reports, for each model, three honest numbers:
  * in-distribution : train on 80% of reference, test on the held-out 20%
  * cross-domain    : train on reference, test on the official test set (shifted)
  * adapted         : train on finetune (same campaign as test), test on test

and then, for the deep model under shift, adds the trust layer the field lacks:
temperature scaling + ECE, and conformal prediction sets with coverage.

    python scripts/run_domain_shift.py --cnn-epochs 40 --train-size 30000
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from raman_ml import uncertainty as U
from raman_ml.augment import SpectralAugment
from raman_ml.datasets import load_bacteria_id
from raman_ml.models import CNNClassifier

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "benchmarks", "results")
PLOTS_DIR = os.path.join(_ROOT, "benchmarks", "plots")


def cap(X, y, n, seed):
    if n is None or n >= len(X):
        return X, y
    idx = train_test_split(np.arange(len(X)), train_size=n, stratify=y,
                           random_state=seed)[0]
    return X[idx], y[idx]


def make_cnn(args):
    aug = SpectralAugment(noise=0.01, offset=0.01, slope=0.01,
                          mult=(0.97, 1.03), shift=3, p=0.5, seed=args.seed)
    return CNNClassifier(n_out=30, epochs=args.cnn_epochs, batch_size=256,
                         arch="resnet", resnet_base=32, label_smoothing=0.05,
                         augment=aug, seed=args.seed)


def evaluate_model(name, factory, Xr, yr, Xf, yf, Xt, yt, seed):
    # in-distribution split of reference
    Xa, Xb, ya, yb = train_test_split(Xr, yr, test_size=0.2, stratify=yr,
                                      random_state=seed)
    m_in = factory(); m_in.fit(Xa, ya)
    acc_in = accuracy_score(yb, m_in.predict(Xb))
    # cross-domain: trained on (subsampled) reference -> test
    m_cross = factory(); m_cross.fit(Xr, yr)
    acc_cross = accuracy_score(yt, m_cross.predict(Xt))
    # adapted: trained on finetune (same campaign as test) -> test
    m_adapt = factory(); m_adapt.fit(Xf, yf)
    acc_adapt = accuracy_score(yt, m_adapt.predict(Xt))
    print(f"{name:>16}  in-dist={acc_in:.3f}  cross-domain={acc_cross:.3f}  "
          f"adapted={acc_adapt:.3f}  (shift gap={acc_in - acc_cross:+.3f})")
    return dict(model=name, in_distribution=acc_in, cross_domain=acc_cross,
                adapted=acc_adapt), m_cross


def trust_layer(cnn_cross, Xf, yf, Xt, yt, alpha, seed):
    """Temperature scaling + conformal sets, calibrated on finetune, eval on test."""
    cal_logits = cnn_cross._raw_predict(Xf)
    test_logits = cnn_cross._raw_predict(Xt)
    T = U.fit_temperature(cal_logits, yf)
    p_cal_ts = U.softmax(cal_logits, T=T)
    p_test_raw = U.softmax(test_logits, T=1.0)
    p_test_ts = U.softmax(test_logits, T=T)
    ece_before = U.expected_calibration_error(p_test_raw, yt)
    ece_after = U.expected_calibration_error(p_test_ts, yt)
    fn, qhat = U.calibrate_conformal_classifier(p_cal_ts, yf, alpha=alpha,
                                                raps=True, seed=seed)
    masks = fn(p_test_ts)
    sm = U.set_metrics(masks, yt)
    print(f"\n  Trust layer (temperature scaling + RAPS conformal, alpha={alpha}):")
    print(f"    temperature T = {T:.2f}")
    print(f"    ECE  before={ece_before:.3f}  after={ece_after:.3f}")
    print(f"    conformal sets: coverage={sm['coverage']:.3f} "
          f"(target {1 - alpha:.2f}), avg set size={sm['avg_set_size']:.2f}/30")
    return dict(temperature=T, ece_before=ece_before, ece_after=ece_after,
                conformal_coverage=sm["coverage"],
                conformal_set_size=sm["avg_set_size"], alpha=alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn-epochs", type=int, default=40)
    ap.add_argument("--train-size", type=int, default=30000)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    Xr, yr, _ = load_bacteria_id("reference")
    Xf, yf, _ = load_bacteria_id("finetune")
    Xt, yt, _ = load_bacteria_id("test")
    Xr, yr = cap(Xr, yr, args.train_size, args.seed)
    print(f"reference={Xr.shape}  finetune={Xf.shape}  test={Xt.shape}\n")

    rows = []
    rows.append(evaluate_model(
        "LogReg", lambda: LogisticRegression(max_iter=300),
        Xr, yr, Xf, yf, Xt, yt, args.seed)[0])
    rows.append(evaluate_model(
        "LinearSVM", lambda: LinearSVC(C=1.0, dual=False, random_state=args.seed),
        Xr, yr, Xf, yf, Xt, yt, args.seed)[0])
    rows.append(evaluate_model(
        "RandomForest", lambda: RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=args.seed),
        Xr, yr, Xf, yf, Xt, yt, args.seed)[0])
    cnn_row, cnn_cross = evaluate_model(
        "1D-ResNet", lambda: make_cnn(args), Xr, yr, Xf, yf, Xt, yt, args.seed)
    rows.append(cnn_row)

    trust = trust_layer(cnn_cross, Xf, yf, Xt, yt, args.alpha, args.seed)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "domain_shift_metrics.csv"), index=False)
    import json
    with open(os.path.join(RESULTS_DIR, "domain_shift_params.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)  # provenance: params -> metrics
    pd.DataFrame([trust]).to_csv(
        os.path.join(RESULTS_DIR, "trust_layer_metrics.csv"), index=False)
    print("\n" + df.to_string(index=False))

    # grouped bar chart: the domain-shift gap and its recovery
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(df)); w = 0.26
    ax.bar(x - w, df["in_distribution"], w, label="in-distribution", color="#4C72B0")
    ax.bar(x, df["cross_domain"], w, label="cross-domain (shift)", color="#C44E52")
    ax.bar(x + w, df["adapted"], w, label="adapted (+finetune)", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(df["model"])
    ax.set_ylabel("accuracy (30-class)"); ax.set_ylim(0, 1)
    ax.set_title("bacteria-ID domain shift: in-distribution vs cross-domain vs adapted")
    ax.legend(fontsize=8)
    for xi, r in zip(x, df.itertuples(), strict=False):
        for dx, v in [(-w, r.in_distribution), (0, r.cross_domain), (w, r.adapted)]:
            ax.text(xi + dx, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "domain_shift.png"), dpi=130)
    print("\nSaved domain_shift.png and metrics to benchmarks/")


if __name__ == "__main__":
    main()
