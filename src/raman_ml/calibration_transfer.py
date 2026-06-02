"""Calibration transfer: make a model trained on one domain work on another.

Distribution shift across instruments / campaigns / conditions is the dominant
real-world failure mode for Raman models (the bacteria-ID reference->test gap in
this repo is a live example). Piecewise Direct Standardization (PDS, Bouveresse
& Massart 1996) learns a banded linear map from a few "transfer standards"
measured in both domains, so spectra from the secondary domain are mapped onto
the primary domain the model was trained on.

This is a classic, dependency-free chemometric method and is absent from every
maintained open-source Raman library.
"""
from __future__ import annotations

import numpy as np


class PiecewiseDirectStandardization:
    """PDS via local windowed least squares (Bouveresse & Massart 1996).

    For each wavenumber i, regress the primary-domain intensity at i on a window
    of secondary-domain intensities [i-h, i+h]; assemble the local coefficients
    into a banded transfer matrix P (and offset) mapping secondary -> primary.
    """

    def __init__(self, half_window: int = 5, ridge: float = 1e-6):
        self.half_window = half_window
        self.ridge = ridge
        self.P_ = None
        self.intercept_ = None

    def fit(self, X_secondary, X_primary):
        """X_* are paired transfer standards (n_standards, n_wavenumbers)."""
        Xs = np.atleast_2d(np.asarray(X_secondary, float))
        Xp = np.atleast_2d(np.asarray(X_primary, float))
        n, L = Xs.shape
        h = self.half_window
        P = np.zeros((L, L))
        intercept = np.zeros(L)
        for i in range(L):
            lo, hi = max(0, i - h), min(L, i + h + 1)
            W = Xs[:, lo:hi]
            A = np.hstack([np.ones((n, 1)), W])  # bias + window
            y = Xp[:, i]
            G = A.T @ A + self.ridge * np.eye(A.shape[1])
            coef = np.linalg.solve(G, A.T @ y)
            intercept[i] = coef[0]
            P[i, lo:hi] = coef[1:]
        self.P_ = P
        self.intercept_ = intercept
        return self

    def transform(self, X_secondary):
        """Map secondary-domain spectra onto the primary domain."""
        Xs = np.atleast_2d(np.asarray(X_secondary, float))
        return Xs @ self.P_.T + self.intercept_


def apply_transfer_if_improves(model, pds, X_secondary_val, y_val,
                               X_secondary_apply, scorer=None):
    """Guarded transfer: apply PDS to ``X_secondary_apply`` only if it improves
    ``model``'s score on a labelled secondary validation set; otherwise return
    the raw secondary spectra unchanged.

    This prevents over-correction: a model that is already shift-robust (e.g.
    low-component PLSR) can be *hurt* by an imperfect transfer, so we only
    transfer where it demonstrably helps. Returns (X_out, used_pds: bool).
    """
    from sklearn.metrics import r2_score
    scorer = scorer or r2_score
    base = scorer(y_val, model.predict(np.asarray(X_secondary_val, float)))
    mapped = scorer(y_val, model.predict(pds.transform(X_secondary_val)))
    use = mapped > base
    out = pds.transform(X_secondary_apply) if use else np.asarray(X_secondary_apply,
                                                                  float)
    return out, bool(use)


def select_transfer_standards(X, n=10, seed=0):
    """Pick n diverse transfer standards via a simple k-center / max-min greedy.

    Returns the chosen row indices. Diverse standards transfer better than
    random ones when only a handful can be measured in both domains.
    """
    X = np.atleast_2d(np.asarray(X, float))
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(X)))]
    d = np.linalg.norm(X - X[idx[0]], axis=1)
    while len(idx) < min(n, len(X)):
        nxt = int(np.argmax(d))
        idx.append(nxt)
        d = np.minimum(d, np.linalg.norm(X - X[nxt], axis=1))
    return np.array(sorted(set(idx)))
