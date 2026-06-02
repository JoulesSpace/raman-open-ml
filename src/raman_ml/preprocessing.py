"""Shared spectral preprocessing for Raman spectra.

Functions here are deliberately framework-agnostic (NumPy only) so they can be
reused by both the classification and quantification pipelines.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve


def als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01,
                 n_iter: int = 10) -> np.ndarray:
    """Asymmetric Least Squares baseline (Eilers & Boelens, 2005).

    Returns the estimated baseline for a single spectrum ``y``.
    ``lam`` controls smoothness, ``p`` the asymmetry (0 < p < 1).
    """
    y = np.asarray(y, dtype=float)
    L = len(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    z = y
    for _ in range(n_iter):
        W.setdiag(w)
        Z = (W + D).tocsc()
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def arpls_baseline(y: np.ndarray, lam: float = 1e5, ratio: float = 0.05,
                   n_iter: int = 50) -> np.ndarray:
    """Asymmetrically reweighted penalised least squares (Baek et al. 2015).

    Generally distorts large peaks less than plain ALS and needs no peak
    detection; weights adapt from the residual distribution automatically.
    """
    y = np.asarray(y, dtype=float)
    L = len(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
    H = lam * D.dot(D.transpose())
    w = np.ones(L)
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve((W + H).tocsc(), w * y)
        d = y - z
        dn = d[d < 0]
        m, s = (dn.mean(), dn.std()) if dn.size else (0.0, 1.0)
        wt = 1.0 / (1.0 + np.exp(2.0 * (d - (2 * s - m)) / (s + 1e-9)))
        if np.linalg.norm(w - wt) / (np.linalg.norm(w) + 1e-9) < ratio:
            w = wt
            break
        w = wt
    return z


def airpls_baseline(y: np.ndarray, lam: float = 1e5, n_iter: int = 15) -> np.ndarray:
    """Adaptive iteratively reweighted PLS baseline (Zhang et al. 2010)."""
    y = np.asarray(y, dtype=float)
    L = len(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
    H = lam * D.dot(D.transpose())
    w = np.ones(L)
    z = y
    for i in range(1, n_iter + 1):
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve((W + H).tocsc(), w * y)
        d = y - z
        dn = d[d < 0]
        if dn.size == 0 or np.abs(dn).sum() < 0.001 * np.abs(y).sum():
            break
        w = np.zeros(L)
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / (np.abs(dn).sum() + 1e-9))
    return z


def modpoly_baseline(y: np.ndarray, degree: int = 5, n_iter: int = 100,
                     tol: float = 1e-3) -> np.ndarray:
    """ModPoly iterative polynomial baseline (Lieber & Mahadevan-Jansen 2003)."""
    y = np.asarray(y, dtype=float)
    x = np.linspace(-1, 1, len(y))
    work = y.copy()
    prev = None
    for _ in range(n_iter):
        coef = np.polyfit(x, work, degree)
        fit = np.polyval(coef, x)
        work = np.minimum(work, fit)
        if prev is not None and np.sum(np.abs(work - prev)) / (np.sum(np.abs(prev)) + 1e-9) < tol:
            break
        prev = work.copy()
    return np.polyval(np.polyfit(x, work, degree), x)


def imodpoly_baseline(y: np.ndarray, degree: int = 5, n_iter: int = 100,
                      tol: float = 1e-3) -> np.ndarray:
    """IModPoly noise-aware iterative polynomial baseline (Zhao et al. 2007)."""
    y = np.asarray(y, dtype=float)
    x = np.linspace(-1, 1, len(y))
    fit = np.polyval(np.polyfit(x, y, degree), x)
    prev_std = None
    work = y.copy()
    for _ in range(n_iter):
        resid = work - fit
        std = resid.std()
        work = np.minimum(work, fit + std)
        coef = np.polyfit(x, work, degree)
        fit = np.polyval(coef, x)
        if prev_std is not None and abs(prev_std - std) / (prev_std + 1e-9) < tol:
            break
        prev_std = std
    return fit


def snip_baseline(y: np.ndarray, n_iter: int = 40) -> np.ndarray:
    """SNIP baseline (Ryan et al. 1988): iterative peak clipping in log-log space."""
    y = np.asarray(y, dtype=float)
    offset = y.min()
    v = np.log(np.log(np.sqrt(y - offset + 1) + 1) + 1)
    L = len(v)
    for m in range(1, n_iter + 1):
        a = v.copy()
        lo = np.arange(L) - m
        hi = np.arange(L) + m
        valid = (lo >= 0) & (hi < L)
        mid = np.where(valid, (np.roll(v, m) + np.roll(v, -m)) / 2.0, v)
        a[valid] = np.minimum(v[valid], mid[valid])
        v = a
    z = (np.exp(np.exp(v) - 1) - 1) ** 2 - 1 + offset
    return z


def rubberband_baseline(y: np.ndarray) -> np.ndarray:
    """Rubberband (convex-hull) baseline: lower convex hull interpolated."""
    from scipy.spatial import ConvexHull
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float)
    pts = np.column_stack([x, y])
    v = ConvexHull(pts).vertices
    v = np.roll(v, -v.argmin())
    v = v[:v.argmax() + 1]            # lower hull from leftmost to rightmost
    return np.interp(x, x[v], y[v])


_BASELINES = {"als": als_baseline, "arpls": arpls_baseline,
              "airpls": airpls_baseline, "modpoly": modpoly_baseline,
              "imodpoly": imodpoly_baseline, "snip": snip_baseline,
              "rubberband": rubberband_baseline}
_LAM_METHODS = {"als", "arpls", "airpls"}


def remove_baseline(X: np.ndarray, method: str = "als", lam: float = 1e5,
                    p: float = 0.01) -> np.ndarray:
    """Row-wise baseline removal.

    ``method`` in {"als", "arpls", "airpls", "modpoly", "imodpoly", "snip",
    "rubberband"}. Penalised-LS methods use ``lam`` (and ``p`` for ALS); the
    polynomial / clipping / hull methods use their own defaults - call them
    directly for finer control.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    fn = _BASELINES[method]
    out = np.empty_like(X)
    for i, row in enumerate(X):
        if method == "als":
            b = fn(row, lam=lam, p=p)
        elif method in _LAM_METHODS:
            b = fn(row, lam=lam)
        else:
            b = fn(row)
        out[i] = row - b
    return out


def remove_cosmic_rays(X: np.ndarray, threshold: float = 7.0,
                       window: int = 3) -> np.ndarray:
    """Whitaker-Hayes cosmic-ray / spike removal (2018).

    Flags points whose successive-difference modified z-score exceeds
    ``threshold`` and replaces them with the local mean of unflagged neighbours.
    Run this FIRST - a single spike corrupts per-spectrum normalisation and
    injects a spurious feature a classifier will latch onto.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    out = X.copy()
    for i, row in enumerate(X):
        d = np.diff(row)
        med = np.median(d)
        mad = np.median(np.abs(d - med)) or 1e-9
        z = 0.6745 * (d - med) / mad
        spikes = np.where(np.abs(z) > threshold)[0]
        for s in spikes:
            for idx in (s, s + 1):
                lo, hi = max(0, idx - window), min(len(row), idx + window + 1)
                nbrs = [j for j in range(lo, hi)
                        if j != idx and j - 1 not in spikes and j not in spikes]
                if nbrs:
                    out[i, idx] = np.mean(row[nbrs])
    return out


def msc(X: np.ndarray, reference: np.ndarray | None = None):
    """Multiplicative scatter correction. Returns (corrected, reference).

    Leakage warning: when ``reference`` is None the reference spectrum is the
    mean of ``X``. In a train/test or CV setting, fit the reference on the
    TRAINING rows only and pass it back to transform the test rows - do not call
    MSC with ``reference=None`` on the full dataset.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    ref = X.mean(0) if reference is None else np.asarray(reference, float)
    out = np.empty_like(X)
    for i, row in enumerate(X):
        b, a = np.polyfit(ref, row, 1)  # row ~ a + b*ref
        out[i] = (row - a) / (b if b != 0 else 1.0)
    return out, ref


def savgol_derivative(X: np.ndarray, window: int = 11, poly: int = 2,
                      deriv: int = 1) -> np.ndarray:
    """Savitzky-Golay smoothing / derivative (deriv 0=smooth, 1, 2)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if window % 2 == 0:
        window += 1
    return savgol_filter(X, window_length=window, polyorder=poly, deriv=deriv,
                         axis=1)


def snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate: per-spectrum mean-centre and scale by std."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Scale each spectrum to unit L2 norm."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return X / norm


def minmax01(X: np.ndarray) -> np.ndarray:
    """Scale each spectrum to the [0, 1] range."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    lo = X.min(axis=1, keepdims=True)
    hi = X.max(axis=1, keepdims=True)
    rng = hi - lo
    rng[rng == 0] = 1.0
    return (X - lo) / rng


def resample_to_grid(shift: np.ndarray, intensity: np.ndarray,
                     grid: np.ndarray) -> np.ndarray:
    """Linearly interpolate a spectrum onto a common wavenumber ``grid``.

    Handles axes stored in either ascending or descending order.
    """
    shift = np.asarray(shift, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    order = np.argsort(shift)
    return np.interp(grid, shift[order], intensity[order])


def preprocess_pipeline(X: np.ndarray, baseline: bool = True,
                        normalize: str = "l2", lam: float = 1e5,
                        p: float = 0.01) -> np.ndarray:
    """Convenience wrapper: optional baseline removal then normalization.

    ``normalize`` is one of {"l2", "snv", "minmax", "none"}.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if baseline:
        X = remove_baseline(X, lam=lam, p=p)
    if normalize == "l2":
        X = l2_normalize(X)
    elif normalize == "snv":
        X = snv(X)
    elif normalize == "minmax":
        X = minmax01(X)
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"unknown normalize mode: {normalize}")
    return X
