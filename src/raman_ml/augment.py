"""On-the-fly spectral augmentation for 1-D Raman spectra.

These are the augmentations repeatedly shown to improve generalisation for
spectral deep learning (Bjerrum et al. 2017 on Raman; SpecAugment-style masking
from speech; mixup from Zhang et al. 2018). All operate on a NumPy batch of
shape (n, length) and return a new array; `SpectralAugment` composes them with
per-call random parameters so a model sees a fresh view every epoch.

Design choices kept deliberate and cheap so they run inside the training loop on
CPU or GPU-fed pipelines:
  * offset / slope  -> simulate residual baseline the preprocessing missed
  * multiply        -> simulate intensity / focus / concentration scale changes
  * additive noise  -> shot / detector noise
  * wavenumber shift-> simulate small calibration / instrument drift
  * peak warp       -> mild non-linear x-axis warp (random smooth displacement)
  * mask            -> SpecAugment-style band dropout (robustness to occluded peaks)
  * mixup           -> convex combinations of pairs (label-aware, returns mixed y)
"""
from __future__ import annotations

import numpy as np


def add_noise(X, sigma=0.01, rng=None):
    rng = rng or np.random.default_rng()
    return X + rng.standard_normal(X.shape).astype(X.dtype) * sigma


def random_offset(X, max_offset=0.02, rng=None):
    rng = rng or np.random.default_rng()
    off = (rng.random((X.shape[0], 1)) * 2 - 1) * max_offset
    return X + off.astype(X.dtype)


def random_slope(X, max_slope=0.02, rng=None):
    rng = rng or np.random.default_rng()
    n, L = X.shape
    ramp = np.linspace(-0.5, 0.5, L, dtype=X.dtype)[None, :]
    sl = ((rng.random((n, 1)) * 2 - 1) * max_slope).astype(X.dtype)
    return X + sl * ramp


def random_multiply(X, low=0.95, high=1.05, rng=None):
    rng = rng or np.random.default_rng()
    fac = rng.uniform(low, high, size=(X.shape[0], 1)).astype(X.dtype)
    return X * fac


def random_shift(X, max_shift=3, rng=None):
    """Integer wavenumber shift (roll) per spectrum, edge-padded."""
    rng = rng or np.random.default_rng()
    out = np.empty_like(X)
    shifts = rng.integers(-max_shift, max_shift + 1, size=X.shape[0])
    for i, s in enumerate(shifts):
        out[i] = np.roll(X[i], s)
        if s > 0:
            out[i, :s] = X[i, 0]
        elif s < 0:
            out[i, s:] = X[i, -1]
    return out


def random_warp(X, n_anchors=5, strength=2.0, rng=None):
    """Smooth non-linear x-axis warp via random anchor displacements."""
    rng = rng or np.random.default_rng()
    n, L = X.shape
    base = np.arange(L, dtype=np.float64)
    anchors = np.linspace(0, L - 1, n_anchors)
    out = np.empty_like(X)
    for i in range(n):
        disp = rng.standard_normal(n_anchors) * strength
        disp[0] = disp[-1] = 0.0
        warp = np.interp(base, anchors, disp)
        out[i] = np.interp(base + warp, base, X[i].astype(np.float64)).astype(X.dtype)
    return out


def random_mask(X, max_bands=2, max_width=40, rng=None):
    """SpecAugment-style: zero out a few random wavenumber bands."""
    rng = rng or np.random.default_rng()
    out = X.copy()
    L = X.shape[1]
    for i in range(X.shape[0]):
        for _ in range(rng.integers(0, max_bands + 1)):
            w = int(rng.integers(1, max_width + 1))
            start = int(rng.integers(0, max(1, L - w)))
            out[i, start:start + w] = 0.0
    return out


def mixup(X, y, alpha=0.2, rng=None, n_classes=None):
    """Mixup (Zhang et al. 2018). Returns mixed X and soft labels.

    If ``n_classes`` is given, y is treated as class indices and one-hot encoded;
    otherwise y is treated as regression targets and mixed directly.
    """
    rng = rng or np.random.default_rng()
    n = X.shape[0]
    lam = rng.beta(alpha, alpha)
    perm = rng.permutation(n)
    Xm = lam * X + (1 - lam) * X[perm]
    if n_classes is not None:
        eye = np.eye(n_classes, dtype=np.float32)
        Y = eye[y]
        Ym = lam * Y + (1 - lam) * Y[perm]
    else:
        Ym = lam * y + (1 - lam) * y[perm]
    return Xm, Ym


class SpectralAugment:
    """Compose augmentations with per-call random magnitudes.

    Call on a batch ``X`` (n, L); returns an augmented copy. Magnitudes are the
    upper bounds; each op samples within them. Disable any op by setting its
    probability to 0. Intended for spectra already normalised to ~unit scale.
    """

    def __init__(self, noise=0.01, offset=0.02, slope=0.02,
                 mult=(0.95, 1.05), shift=3, warp=0.0, mask_prob=0.0,
                 p=0.5, seed=0):
        self.noise = noise
        self.offset = offset
        self.slope = slope
        self.mult = mult
        self.shift = shift
        self.warp = warp
        self.mask_prob = mask_prob
        self.p = p
        self.rng = np.random.default_rng(seed)

    def _maybe(self):
        return self.rng.random() < self.p

    def __call__(self, X):
        X = np.asarray(X)
        if self.mult and self._maybe():
            X = random_multiply(X, self.mult[0], self.mult[1], self.rng)
        if self.slope and self._maybe():
            X = random_slope(X, self.slope, self.rng)
        if self.offset and self._maybe():
            X = random_offset(X, self.offset, self.rng)
        if self.shift and self._maybe():
            X = random_shift(X, self.shift, self.rng)
        if self.warp and self._maybe():
            X = random_warp(X, strength=self.warp, rng=self.rng)
        if self.mask_prob and self.rng.random() < self.mask_prob:
            X = random_mask(X, rng=self.rng)
        if self.noise and self._maybe():
            X = add_noise(X, self.noise, self.rng)
        return X
