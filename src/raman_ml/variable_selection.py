"""Wavenumber (variable) selection for small-sample quantification.

Full-spectrum PLS overfits when n << p (here n=48, p=1000). Selecting the
informative wavenumbers measurably lowers prediction error. We provide:

* VIP (Variable Importance in Projection) scores from a fitted PLS model -
  the standard, cheap importance measure (Wold et al.; Chong & Jun 2005),
* a thin selector that keeps variables above a VIP threshold (1.0 is the
  conventional cut) or a top-k set.

CARS (competitive adaptive reweighted sampling) is the stronger but heavier
alternative; VIP is the high-impact, low-effort first step.
"""
from __future__ import annotations

import numpy as np


def vip_scores(pls):
    """VIP scores for a fitted sklearn PLSRegression.

    VIP_j = sqrt( p * sum_a [ SS_a * (w_aj / ||w_a||)^2 ] / sum_a SS_a ),
    where SS_a is the variance of the response explained by component a.
    """
    t = pls.x_scores_              # (n, A)
    w = pls.x_weights_             # (p, A)
    q = pls.y_loadings_            # (y_dim, A)
    p, A = w.shape
    # response variance explained per component
    ss = np.array([(t[:, a] ** 2).sum() * (q[:, a] ** 2).sum() for a in range(A)])
    wnorm = w / (np.linalg.norm(w, axis=0, keepdims=True) + 1e-12)
    vip = np.sqrt(p * ((wnorm ** 2) * ss[None, :]).sum(axis=1) / (ss.sum() + 1e-12))
    return vip


def select_by_vip(pls, threshold=1.0, top_k=None):
    """Return indices of selected variables from a fitted PLS.

    With ``top_k`` set, keep the k highest-VIP variables; otherwise keep all
    variables with VIP >= ``threshold``.
    """
    vip = vip_scores(pls)
    if top_k is not None:
        return np.sort(np.argsort(-vip)[:top_k])
    idx = np.where(vip >= threshold)[0]
    return idx if idx.size else np.argsort(-vip)[:max(1, len(vip) // 10)]


class VIPSelectedPLSR:
    """PLSR with leakage-safe VIP variable selection.

    Fits a PLS on the training data, keeps the top-k VIP wavenumbers, and refits
    PLS on that subset. Because everything happens inside ``fit``, it is safe to
    use directly inside a cross-validation loop. Returns 1-D predictions.
    """

    def __init__(self, n_components=3, top_k=400):
        from sklearn.cross_decomposition import PLSRegression
        self._PLS = PLSRegression
        self.n_components = n_components
        self.top_k = top_k
        self.idx_ = None
        self.model_ = None

    def fit(self, X, y):
        X = np.asarray(X, float)
        screen = self._PLS(n_components=self.n_components).fit(X, y)
        self.idx_ = select_by_vip(screen, top_k=min(self.top_k, X.shape[1]))
        self.model_ = self._PLS(n_components=self.n_components).fit(X[:, self.idx_], y)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, float)[:, self.idx_]).ravel()
