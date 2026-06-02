"""Distribution-free uncertainty quantification for spectroscopy models.

No maintained open-source Raman package ships this; it lives only in scattered
2023-2026 papers. We implement the core, framework-agnostic pieces in NumPy:

* split-conformal regression intervals (Vovk; Lei et al. 2018) with finite-sample
  marginal coverage,
* conformalised quantile regression width (CQR, Romano et al. 2019) when a model
  exposes lower/upper quantile predictions,
* APS / RAPS conformal prediction *sets* for classification (Romano 2020;
  Angelopoulos et al. 2021) with coverage and an abstain signal,
* temperature scaling (Guo et al. 2017) and the Expected Calibration Error.

Everything takes a held-out calibration split; nothing peeks at the test labels.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Regression: split conformal intervals                                        #
# --------------------------------------------------------------------------- #
def conformal_interval(cal_y, cal_pred, test_pred, alpha=0.1):
    """Split-conformal absolute-residual intervals.

    Returns (lo, hi) for each test point such that, marginally,
    P(y in [lo, hi]) >= 1 - alpha (exchangeability assumed).
    """
    cal_y = np.asarray(cal_y, float)
    scores = np.abs(cal_y - np.asarray(cal_pred, float))
    n = len(scores)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    q = min(q, 1.0)
    qhat = np.quantile(scores, q, method="higher")
    test_pred = np.asarray(test_pred, float)
    return test_pred - qhat, test_pred + qhat


def cqr_interval(cal_y, cal_lo, cal_hi, test_lo, test_hi, alpha=0.1):
    """Conformalised quantile regression (Romano et al. 2019).

    cal_lo/cal_hi are a model's lower/upper quantile predictions on the
    calibration set; test_lo/test_hi on the test set. Adjusts width for coverage.
    """
    cal_y = np.asarray(cal_y, float)
    E = np.maximum(np.asarray(cal_lo, float) - cal_y,
                   cal_y - np.asarray(cal_hi, float))
    n = len(E)
    q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    qhat = np.quantile(E, q, method="higher")
    return np.asarray(test_lo, float) - qhat, np.asarray(test_hi, float) + qhat


def jackknife_plus_interval(estimator_factory, X, y, X_test, alpha=0.1):
    """Jackknife+ prediction intervals (Barber et al. 2021).

    More reliable than split conformal for small n: refits the model leaving each
    point out, builds intervals from leave-one-out residuals. ``estimator_factory``
    returns a fresh unfitted estimator with ``fit``/``predict``. Returns (lo, hi).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    X_test = np.asarray(X_test, float)
    n = len(X)
    loo_res = np.empty(n)
    test_preds = np.empty((n, len(X_test)))
    for i in range(n):
        mask = np.ones(n, bool)
        mask[i] = False
        m = estimator_factory().fit(X[mask], y[mask])
        loo_res[i] = abs(y[i] - np.ravel(m.predict(X[i:i + 1]))[0])
        test_preds[i] = np.ravel(m.predict(X_test))
    # finite-sample jackknife+ levels (Barber et al. 2021): use the
    # ceil((1-alpha)(n+1))-th order statistic, not the plain (1-alpha) quantile,
    # else the interval under-covers for small n.
    q_hi = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    q_lo = max(np.floor(alpha * (n + 1)) / n, 0.0)
    lo = np.quantile(test_preds - loo_res[:, None], q_lo, axis=0, method="lower")
    hi = np.quantile(test_preds + loo_res[:, None], q_hi, axis=0, method="higher")
    return lo, hi


def interval_metrics(y, lo, hi):
    """Empirical coverage and mean interval width."""
    y = np.asarray(y, float)
    cov = float(np.mean((y >= lo) & (y <= hi)))
    width = float(np.mean(np.asarray(hi) - np.asarray(lo)))
    return {"coverage": cov, "mean_width": width}


# --------------------------------------------------------------------------- #
# Classification: conformal prediction sets (APS / RAPS)                        #
# --------------------------------------------------------------------------- #
def _aps_scores(probs, labels, rng, reg=None):
    """APS/RAPS nonconformity score for the true label of each row."""
    n, K = probs.shape
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)
    rank = np.argmax(order == labels[:, None], axis=1)  # rank of true label
    u = rng.random(n)
    # cumulative prob up to (and randomised within) the true label's rank
    prev = cumsum[np.arange(n), rank] - sorted_p[np.arange(n), rank]
    score = prev + u * sorted_p[np.arange(n), rank]
    if reg is not None:
        k_reg, lam = reg
        score = score + lam * np.maximum(0, (rank + 1) - k_reg)
    return score


def calibrate_conformal_classifier(cal_probs, cal_labels, alpha=0.1,
                                   raps=True, k_reg=3, lam=0.05, seed=0):
    """Return a function mapping test probabilities -> boolean inclusion masks.

    APS (raps=False) or RAPS (raps=True, regularises toward smaller sets).
    Guarantees marginal coverage >= 1 - alpha on exchangeable data.
    """
    rng = np.random.default_rng(seed)
    cal_probs = np.asarray(cal_probs, float)
    cal_labels = np.asarray(cal_labels, int)
    reg = (k_reg, lam) if raps else None
    scores = _aps_scores(cal_probs, cal_labels, rng, reg=reg)
    n = len(scores)
    q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    qhat = np.quantile(scores, q, method="higher")

    def predict_set(test_probs):
        test_probs = np.asarray(test_probs, float)
        m, K = test_probs.shape
        order = np.argsort(-test_probs, axis=1)
        sorted_p = np.take_along_axis(test_probs, order, axis=1)
        cumsum = np.cumsum(sorted_p, axis=1)
        ranks = np.arange(K)[None, :]
        prev = cumsum - sorted_p
        u = rng.random((m, K))
        s = prev + u * sorted_p
        if reg is not None:
            s = s + lam * np.maximum(0, (ranks + 1) - k_reg)
        include_sorted = s <= qhat
        include_sorted[:, 0] = True  # always keep the top-1 (non-empty sets)
        mask = np.zeros((m, K), bool)
        np.put_along_axis(mask, order, include_sorted, axis=1)
        return mask

    return predict_set, qhat


def set_metrics(masks, y):
    """Coverage and average set size for conformal classification sets."""
    y = np.asarray(y, int)
    covered = masks[np.arange(len(y)), y]
    return {"coverage": float(covered.mean()),
            "avg_set_size": float(masks.sum(1).mean())}


# --------------------------------------------------------------------------- #
# Calibration: temperature scaling + ECE                                       #
# --------------------------------------------------------------------------- #
def fit_temperature(logits, labels, lr=0.01, iters=300):
    """Optimise a single temperature T to minimise NLL (Guo et al. 2017).

    Pure-NumPy gradient descent on T>0. Returns the scalar temperature.
    """
    logits = np.asarray(logits, float)
    labels = np.asarray(labels, int)
    logT = 0.0
    n = len(labels)
    for _ in range(iters):
        T = np.exp(logT)
        z = logits / T
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        # d NLL / d logT
        grad_z = p.copy()
        grad_z[np.arange(n), labels] -= 1.0
        dlogit_dlogT = -logits / T
        grad = float(np.sum(grad_z * dlogit_dlogT) / n)
        logT -= lr * grad
    return float(np.exp(logT))


def softmax(logits, T=1.0):
    z = np.asarray(logits, float) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def expected_calibration_error(probs, labels, n_bins=15):
    """ECE: gap between confidence and accuracy across confidence bins."""
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, int)
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)
