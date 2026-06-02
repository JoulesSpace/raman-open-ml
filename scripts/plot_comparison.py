"""Algorithm-comparison plots from the benchmark CSVs (acoustic-drone-detection
style): grouped metric bars + a cost-vs-quality (Pareto) scatter.

Reads benchmarks/results/{classification,quantification}_metrics.csv (produced by
the runner scripts) and writes:
  plots/classification_metrics_bar.png   accuracy / balanced-acc / macro-F1 per model
  plots/classification_cost_quality.png  fit time (log x) vs accuracy, annotated
  plots/quantification_cost_quality.png  CV time (log x) vs R2, annotated

    python scripts/plot_comparison.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "benchmarks", "results")
PLOTS = os.path.join(ROOT, "benchmarks", "plots")
ACCENT, GOOD, MUTE, INK = "#1f77b4", "#2a9d8f", "#8a94a6", "#12263a"


def _pareto_front(x, y, maximize_y=True):
    """Indices on the lower-cost / higher-quality Pareto frontier."""
    order = np.argsort(x)
    best, front = -np.inf, []
    for i in order:
        if (y[i] > best) if maximize_y else (y[i] < best):
            best = y[i]
            front.append(i)
    return front


def metrics_bar(df, metrics, title, path):
    df = df.sort_values(metrics[0], ascending=False)
    names = df["model"].tolist()
    n, b = len(names), len(metrics)
    width = 0.8 / b
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * n), 4.5))
    for i, m in enumerate(metrics):
        xs = [g + (i - (b - 1) / 2) * width for g in range(n)]
        yerr = df[f"{m}_std"].to_numpy() if f"{m}_std" in df else None
        ax.bar(xs, df[m].to_numpy(), width=width, label=m.replace("_", " "),
               yerr=yerr, capsize=2, error_kw={"elinewidth": 0.7})
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("score"); ax.set_title(title)
    ax.legend(loc="lower right", ncol=b, fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def cost_quality(df, cost_col, quality_col, qlabel, title, path):
    x = np.maximum(df[cost_col].to_numpy(float), 1e-2)
    y = df[quality_col].to_numpy(float)
    names = df["model"].tolist()
    fig, ax = plt.subplots(figsize=(7, 5.2))
    front = _pareto_front(x, y)
    ax.plot(x[front], y[front], "-", color=GOOD, lw=1.2, alpha=0.7, zorder=1,
            label="Pareto frontier")
    for i, name in enumerate(names):
        on = i in front
        ax.scatter(x[i], y[i], s=70, color=GOOD if on else ACCENT,
                   edgecolor=INK, lw=0.5, zorder=3)
        ax.annotate(name, (x[i], y[i]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    # Pad the log x-range so point labels (some are long, e.g. the ensemble) fit.
    ax.set_xlim(x.min() * 0.5, x.max() * 6)
    ax.set_xlabel(f"{cost_col} - training cost, seconds (log scale)")
    ax.set_ylabel(qlabel); ax.set_title(title)
    ax.grid(alpha=0.3, which="both"); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    os.makedirs(PLOTS, exist_ok=True)
    clf_csv = os.path.join(RESULTS, "classification_metrics.csv")
    if os.path.exists(clf_csv):
        df = pd.read_csv(clf_csv)
        metrics_bar(df, ["accuracy", "balanced_accuracy", "macro_f1"],
                    "Classification metrics by algorithm",
                    os.path.join(PLOTS, "classification_metrics_bar.png"))
        cost_quality(df, "fit_seconds", "accuracy", "test accuracy",
                     "Classification: cost vs quality (upper-left is best)",
                     os.path.join(PLOTS, "classification_cost_quality.png"))
        print("wrote classification_metrics_bar.png, classification_cost_quality.png")
    quant_csv = os.path.join(RESULTS, "quantification_metrics.csv")
    if os.path.exists(quant_csv):
        df = pd.read_csv(quant_csv)
        cost_quality(df, "time_s", "R2", "R$^2$ (CV)",
                     "Quantification: cost vs quality (upper-left is best)",
                     os.path.join(PLOTS, "quantification_cost_quality.png"))
        print("wrote quantification_cost_quality.png")


if __name__ == "__main__":
    main()
