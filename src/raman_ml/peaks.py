"""Peak-feature extraction: turn a 1000-point spectrum into a few interpretable
band features (position, area, height, FWHM) and ratios.

Why this matters (peak-fitting libraries raman-fitting / SpectraFit / PeakFit all
do versions of this): peak features are physically
meaningful, low-dimensional (great for small-data sklearn models and to
regularise a CNN), and area-ratios (analyte band / reference band) are the
classic intensity-calibration-robust quantification predictor.

We use a fixed *band registry* discovered on the training mean spectrum so the
feature columns mean the same band for every sample (essential for ML). No
external dependency - scipy.signal only (lmfit-style curve fitting is optional
and not required for robust area/FWHM features).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_widths


class PeakFeatureExtractor:
    """Discover bands on the training mean spectrum, then extract per-band
    [area, height, position-shift, FWHM] features for any spectrum.

    sklearn-style: ``fit(X)`` finds the bands, ``transform(X)`` returns an
    (n_samples, n_bands * 4) feature matrix. ``feature_names_`` documents columns.
    """

    def __init__(self, n_bands=20, window=8, prominence=0.01, features=("area",
                 "height", "shift", "fwhm")):
        self.n_bands = n_bands
        self.window = window
        self.prominence = prominence
        self.features = features
        self.bands_ = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        mean = X.mean(0)
        mean_n = (mean - mean.min()) / (np.ptp(mean) + 1e-12)
        peaks, props = find_peaks(mean_n, prominence=self.prominence)
        if len(peaks) == 0:
            peaks = np.array([int(np.argmax(mean))])
            props = {"prominences": np.array([1.0])}
        order = np.argsort(-props["prominences"])[:self.n_bands]
        self.bands_ = np.sort(peaks[order])
        names = []
        for b in self.bands_:
            names += [f"band{b}_{f}" for f in self.features]
        self.feature_names_ = names
        return self

    def transform(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.empty((len(X), len(self.bands_) * len(self.features)))
        w = self.window
        for i, row in enumerate(X):
            feats = []
            for b in self.bands_:
                lo, hi = max(0, b - w), min(len(row), b + w + 1)
                seg = row[lo:hi]
                local = lo + int(np.argmax(seg))
                area = float(np.trapezoid(seg - seg.min()))
                height = float(row[local] - seg.min())
                shift = float(local - b)
                try:
                    fw = peak_widths(row, [local], rel_height=0.5)[0][0]
                except Exception:  # noqa: BLE001
                    fw = float(hi - lo)
                vals = {"area": area, "height": height, "shift": shift,
                        "fwhm": float(fw)}
                feats += [vals[f] for f in self.features]
            out[i] = feats
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def band_area(spectrum, wavenumbers, lo, hi):
    """Integrated area of a spectrum between two wavenumbers (trapezoid)."""
    spectrum = np.asarray(spectrum, float)
    wn = np.asarray(wavenumbers, float)
    order = np.argsort(wn)
    wn, spectrum = wn[order], spectrum[order]
    mask = (wn >= lo) & (wn <= hi)
    if mask.sum() < 2:
        return 0.0
    seg = spectrum[mask]
    return float(np.trapezoid(seg - seg.min(), wn[mask]))


def area_ratio(spectrum, wavenumbers, analyte, reference):
    """Analyte-band / reference-band area ratio - the classic SERS quantification
    feature (robust to absolute intensity drift). ``analyte``/``reference`` are
    (lo, hi) wavenumber windows.
    """
    a = band_area(spectrum, wavenumbers, *analyte)
    r = band_area(spectrum, wavenumbers, *reference)
    return a / r if r else np.nan
