"""Publication-quality preprocessing figures (DeepeR-style method comparison).

Two figures, both on a real polystyrene spectrum (raw SERS, real fluorescence
baseline) so the steps are visible:

  plots/preprocessing_cascade.png    one spectrum through the whole pipeline:
                                     raw -> cosmic-ray removed -> baseline
                                     corrected -> SNV -> Savitzky-Golay 1st deriv
  plots/baseline_comparison.png      the five baseline estimators overlaid on the
                                     same raw spectrum (left), and the resulting
                                     corrected spectra (right) - the Raman analogue
                                     of DeepeR's "input vs method-A vs method-B" panel

    python scripts/plot_preprocessing_showcase.py
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from raman_ml.datasets import load_polystyrene  # noqa: E402
from raman_ml.preprocessing import (  # noqa: E402
    airpls_baseline,
    als_baseline,
    arpls_baseline,
    modpoly_baseline,
    remove_cosmic_rays,
    savgol_derivative,
    snip_baseline,
    snv,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = os.path.join(ROOT, "benchmarks", "plots")
ACCENT, GOOD, INK, MUTE = "#1f77b4", "#2a9d8f", "#12263a", "#8a94a6"


def _pick_spectrum():
    """A high-concentration polystyrene spectrum (clearest peaks + real baseline)."""
    X, _SD, conc, _sid, grid = load_polystyrene()
    i = int(np.argmax(conc))
    return X[i], grid


def cascade(y, wn, path):
    cr = remove_cosmic_rays(y[None])[0]
    base = arpls_baseline(cr)
    corr = cr - base
    normed = snv(corr[None])[0]
    deriv = savgol_derivative(corr[None], window=11, poly=2, deriv=1)[0]

    stages = [
        ("1. Raw mean spectrum", y, INK, None),
        ("2. Cosmic-ray removed (Whitaker-Hayes)", cr, ACCENT, None),
        ("3. Baseline corrected (arPLS)", corr, GOOD, ("baseline", base, cr)),
        ("4. SNV normalised", normed, "#e76f51", None),
        ("5. Savitzky-Golay 1st derivative", deriv, "#9b5de5", None),
    ]
    fig, axes = plt.subplots(len(stages), 1, figsize=(8.5, 10), sharex=True)
    for ax, (title, sig, color, overlay) in zip(axes, stages, strict=True):
        if overlay is not None:
            _, base_curve, raw_curve = overlay
            ax.plot(wn, raw_curve, color=MUTE, lw=0.8, alpha=0.6, label="pre-correction")
            ax.plot(wn, base_curve, color="#d62728", lw=1.0, ls="--", label="estimated baseline")
        ax.plot(wn, sig, color=color, lw=1.0)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.25)
        ax.set_ylabel("intensity (a.u.)", fontsize=8)
        if overlay is not None:
            ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("Raman shift (cm$^{-1}$)")
    fig.suptitle("Preprocessing cascade on one polystyrene spectrum", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def baseline_comparison(y, wn, path):
    cr = remove_cosmic_rays(y[None])[0]
    methods = [
        ("ALS", als_baseline(cr), ACCENT),
        ("arPLS", arpls_baseline(cr), GOOD),
        ("airPLS", airpls_baseline(cr), "#e76f51"),
        ("ModPoly", modpoly_baseline(cr), "#9b5de5"),
        ("SNIP", snip_baseline(cr), "#f4a261"),
    ]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    axL.plot(wn, cr, color=INK, lw=1.1, label="raw spectrum")
    for name, base, color in methods:
        axL.plot(wn, base, color=color, lw=1.0, ls="--", label=f"{name} baseline")
    axL.set_title("Estimated baselines (five algorithms)", fontsize=11, loc="left")
    axL.set_xlabel("Raman shift (cm$^{-1}$)")
    axL.set_ylabel("intensity (a.u.)")
    axL.grid(alpha=0.25)
    axL.legend(loc="upper right", fontsize=8)

    for name, base, color in methods:
        axR.plot(wn, cr - base, color=color, lw=1.0, label=name)
    axR.axhline(0, color=MUTE, lw=0.6)
    axR.set_title("Baseline-corrected result per method", fontsize=11, loc="left")
    axR.set_xlabel("Raman shift (cm$^{-1}$)")
    axR.set_ylabel("intensity (a.u.)")
    axR.grid(alpha=0.25)
    axR.legend(loc="upper right", fontsize=8)
    fig.suptitle("Baseline-correction method comparison (polystyrene)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    os.makedirs(PLOTS, exist_ok=True)
    y, wn = _pick_spectrum()
    cascade(y, wn, os.path.join(PLOTS, "preprocessing_cascade.png"))
    baseline_comparison(y, wn, os.path.join(PLOTS, "baseline_comparison.png"))
    print("wrote preprocessing_cascade.png, baseline_comparison.png")


if __name__ == "__main__":
    main()
