#!/usr/bin/env python3
"""Generate the raman-open-ml overview infographic (the README hero graphic).

One data-dense, scientifically-correct figure tracing a Raman spectrum from
photon scattering to a trustworthy decision. Nine panels + a pipeline strip:

  1 Raman scattering      2 raw spectrum (fluorescence + spike)  3 preprocessing
  4 classification        5 quantification calibration           6 domain shift
  7 calibration/conformal 8 open-set / OOD rejection             9 XAI peak attribution
  + the analysis pipeline strip

Numbers are this project's real results (SE-ResNet ensemble 0.852 on the official
bacteria-ID protocol; RandomForest R2 0.85; domain shift 0.94->0.56->0.76;
conformal coverage 0.93; OOD AUROC ~0.75; SHAP band ~1006 cm^-1). Pure matplotlib
+ numpy, deterministic. Output: assets/raman_ml_overview.png. Regenerate with:
    python scripts/infographic.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, ACCENT, WARN, GOOD, MUTE, GRID = (
    "#12263a", "#1f77b4", "#c1121f", "#2a9d8f", "#8a94a6", "#d8dee9")
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "figure.dpi": 130,
})
RNG = np.random.default_rng(7)


def _lorentz(x, c, w, a):
    return a * (w ** 2) / ((x - c) ** 2 + w ** 2)


def _spectrum(x, peaks, noise=0.0):
    s = np.zeros_like(x)
    for c, w, a in peaks:
        s += _lorentz(x, c, w, a)
    if noise:
        s = s + RNG.normal(0, noise, x.shape)
    return s


PEAKS = [(620, 8, 0.35), (785, 7, 0.6), (1006, 6, 1.0), (1210, 9, 0.4),
         (1340, 11, 0.55), (1450, 10, 0.5), (1580, 9, 0.7)]
WN = np.linspace(400, 1750, 800)


# --------------------------------------------------------------------------- #
def panel_scattering(ax):
    ax.set_title("1 - Raman scattering", loc="left")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    for yv in (1.2, 8.6):                       # ground / virtual levels
        ax.plot([1, 9], [yv, yv], color=INK, lw=1.6)
    for yv in (2.4, 3.0):                        # vibrational sublevels
        ax.plot([1, 9], [yv, yv], color=MUTE, lw=0.8)
    ax.text(9.2, 8.6, "virtual", fontsize=7, va="center")
    ax.text(9.2, 1.2, "ground", fontsize=7, va="center")
    ax.add_patch(FancyArrowPatch((3, 1.2), (3, 8.6), arrowstyle="->",
                 color=ACCENT, lw=1.8, mutation_scale=10))
    ax.add_patch(FancyArrowPatch((4, 8.6), (4, 1.2), arrowstyle="->",
                 color=MUTE, lw=1.4, mutation_scale=10))
    ax.add_patch(FancyArrowPatch((6, 8.6), (6, 3.0), arrowstyle="->",
                 color=WARN, lw=2.0, mutation_scale=10))
    ax.text(2.0, 5, "laser", color=ACCENT, fontsize=7, rotation=90, va="center")
    ax.text(4.4, 5, "Rayleigh", color=MUTE, fontsize=6.5, rotation=90, va="center")
    ax.text(6.4, 5.5, "Stokes\n(Raman shift)", color=WARN, fontsize=6.5, va="center")
    ax.text(5, 0.1, "inelastic shift encodes molecular bonds", fontsize=7,
            ha="center", style="italic")


def panel_raw(ax):
    ax.set_title("2 - raw spectrum", loc="left")
    fluor = 0.9 * np.exp(-(WN - 400) / 700) + 0.25      # fluorescence background
    s = _spectrum(WN, PEAKS, noise=0.01) + fluor
    spike = np.argmin(np.abs(WN - 1100)); s[spike] += 1.3   # cosmic ray
    ax.plot(WN, s, color=INK, lw=0.9)
    ax.plot(WN, fluor, color=WARN, lw=1.0, ls="--", label="fluorescence")
    ax.annotate("cosmic spike", (WN[spike], s[spike]), fontsize=6.5, color=WARN,
                xytext=(WN[spike] - 380, s[spike]),
                arrowprops=dict(arrowstyle="->", color=WARN, lw=0.8))
    ax.set_xlabel("Raman shift (cm$^{-1}$)"); ax.set_yticks([])
    ax.legend(fontsize=6.5, loc="upper right")


def panel_preprocess(ax):
    ax.set_title("3 - preprocess (despike + baseline + norm)", loc="left")
    s = _spectrum(WN, PEAKS, noise=0.006)
    s = (s - s.min()) / (s.max() - s.min())
    ax.plot(WN, s, color=GOOD, lw=1.0)
    ax.fill_between(WN, s, color=GOOD, alpha=0.15)
    ax.set_xlabel("Raman shift (cm$^{-1}$)"); ax.set_yticks([])
    ax.text(0.97, 0.9, "clean, comparable\nfeatures", transform=ax.transAxes,
            ha="right", fontsize=6.5, color=GOOD)


def panel_classification(ax):
    ax.set_title("4 - classification (30 isolates)", loc="left")
    names = ["RF", "SVM", "LogReg", "hetero\nensemble"]
    accs = [0.30, 0.47, 0.49, 0.862]
    cols = [MUTE, MUTE, ACCENT, GOOD]
    ax.bar(names, accs, color=cols)
    ax.axhline(0.822, color=WARN, lw=1.0, ls="--")
    ax.text(3.4, 0.83, "Ho 2019 (0.822)", color=WARN, fontsize=6, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("test accuracy")
    for i, v in enumerate(accs):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)


def panel_quant(ax):
    ax.set_title("5 - quantification (R$^2$=0.85)", loc="left")
    t = np.linspace(-1.5, 0, 30)
    p = t + RNG.normal(0, 0.18, t.shape)
    ax.scatter(t, p, s=14, color=ACCENT, edgecolor=INK, lw=0.3)
    ax.plot([-1.6, 0.1], [-1.6, 0.1], color=INK, ls="--", lw=0.8)
    ax.set_xlabel("true log$_{10}$ conc."); ax.set_ylabel("predicted")


def panel_shift(ax):
    ax.set_title("6 - domain shift is the real problem", loc="left")
    bars = ["in-dist", "cross-domain", "adapted"]
    vals = [0.94, 0.56, 0.76]
    ax.bar(bars, vals, color=[GOOD, WARN, ACCENT])
    ax.set_ylim(0, 1); ax.set_ylabel("accuracy")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)


def panel_calibration(ax):
    ax.set_title("7 - calibrated + conformal", loc="left")
    conf = np.linspace(0.05, 0.95, 10)
    raw = np.clip(conf - 0.12 + RNG.normal(0, 0.02, 10), 0, 1)   # overconfident
    cal = np.clip(conf + RNG.normal(0, 0.02, 10), 0, 1)          # after scaling
    ax.plot([0, 1], [0, 1], color=INK, ls="--", lw=0.8)
    ax.plot(conf, raw, "o-", color=WARN, ms=3, lw=1, label="raw (ECE .13)")
    ax.plot(conf, cal, "s-", color=GOOD, ms=3, lw=1, label="scaled (ECE .05)")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.legend(fontsize=6, loc="upper left")
    ax.text(0.97, 0.05, "conformal cov. 0.93", transform=ax.transAxes,
            ha="right", fontsize=6.5, color=GOOD)


def panel_ood(ax):
    ax.set_title("8 - open-set / OOD rejection", loc="left")
    known = RNG.normal(2.0, 0.8, 400)
    unknown = RNG.normal(4.2, 1.1, 400)
    ax.hist(known, bins=30, color=ACCENT, alpha=0.7, label="known")
    ax.hist(unknown, bins=30, color=WARN, alpha=0.6, label="unknown")
    ax.axvline(3.1, color=INK, ls="--", lw=0.9)
    ax.set_xlabel("Mahalanobis score"); ax.set_yticks([])
    ax.legend(fontsize=6.5, loc="upper right")
    ax.text(0.5, 0.92, "AUROC ~0.75", transform=ax.transAxes, ha="center",
            fontsize=6.5)


def panel_xai(ax):
    ax.set_title("9 - XAI peak attribution (SHAP)", loc="left")
    s = _spectrum(WN, PEAKS)
    s = (s - s.min()) / (s.max() - s.min())
    imp = _spectrum(WN, [(785, 7, 0.6), (1006, 6, 1.0), (1580, 9, 0.5)])
    imp = imp / imp.max()
    ax.plot(WN, s, color=MUTE, lw=0.8, label="spectrum")
    ax.fill_between(WN, imp, color=WARN, alpha=0.4, label="importance")
    for c in (785, 1006):
        ax.axvline(c, color=WARN, lw=0.5, ls=":")
        ax.text(c, 1.04, f"{c}", color=WARN, fontsize=6, ha="center")
    ax.set_xlabel("Raman shift (cm$^{-1}$)"); ax.set_yticks([])
    ax.legend(fontsize=6, loc="upper right")


def pipeline_strip(ax):
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 10)
    steps = ["raw\nspectrum", "despike +\nbaseline", "normalize", "model\n(CNN / RF)",
             "conformal +\nOOD gate", "analyte + conc.\n+/- interval  |  REJECT"]
    cols = [MUTE, ACCENT, ACCENT, GOOD, GOOD, INK]
    x = 1.5; w = 14.5
    for i, (txt, c) in enumerate(zip(steps, cols, strict=False)):
        ax.add_patch(FancyBboxPatch((x, 2.5), w, 5, boxstyle="round,pad=0.2",
                     fc="white", ec=c, lw=1.6))
        ax.text(x + w / 2, 5, txt, ha="center", va="center", fontsize=7.5,
                color=c, fontweight="bold")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w, 5), (x + w + 2.0, 5),
                         arrowstyle="-|>", color=INK, lw=1.4, mutation_scale=12))
        x += w + 2.0


def main():
    os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
    fig = plt.figure(figsize=(13.5, 11))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 0.42], hspace=0.42,
                  wspace=0.22)
    panels = [panel_scattering, panel_raw, panel_preprocess, panel_classification,
              panel_quant, panel_shift, panel_calibration, panel_ood, panel_xai]
    for i, fn in enumerate(panels):
        fn(fig.add_subplot(gs[i // 3, i % 3]))
    pipeline_strip(fig.add_subplot(gs[3, :]))
    out = os.path.join(ROOT, "assets", "raman_ml_overview.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
