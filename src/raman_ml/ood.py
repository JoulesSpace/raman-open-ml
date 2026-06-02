"""Out-of-distribution / open-set detection for spectral classifiers.

The stated #1 clinical limitation for Raman classifiers (Lebron et al. 2024) is
that a closed-set softmax model labels an *unknown* species as a known one with
high confidence. These detectors produce a scalar "novelty" score so unseen
analytes/isolates can be rejected. All are post-hoc (no retraining):

* MSP   - maximum softmax probability (Hendrycks & Gimpel 2017) baseline
* Energy- -logsumexp(logits) (Liu et al. 2020), better-calibrated than MSP
* Mahalanobis - class-conditional Gaussian distance in feature space
  (Lee et al. 2018), the strongest simple detector

Higher score = more likely OOD for all three (energy/Mahalanobis are negated /
distance-based; MSP is negated max-prob). Evaluate with AUROC and FPR@95%TPR.
"""
from __future__ import annotations

import numpy as np


def msp_score(probs):
    """Max-softmax-probability OOD score (higher = more OOD)."""
    return 1.0 - np.asarray(probs, float).max(1)


def energy_score(logits, T=1.0):
    """Energy OOD score E = -T*logsumexp(logits/T) (higher = more OOD).

    In-distribution inputs have large logits -> large logsumexp -> very negative
    (low) energy; OOD inputs have smaller logits -> higher energy.
    """
    z = np.asarray(logits, float) / T
    m = z.max(1, keepdims=True)
    lse = m.squeeze(1) + np.log(np.exp(z - m).sum(1))
    return -T * lse


class MahalanobisOOD:
    """Class-conditional Gaussian detector in a feature space (Lee et al. 2018).

    Fit on in-distribution features + labels; score = min over classes of the
    Mahalanobis distance to the class mean using a shared covariance.
    """

    def __init__(self, shrinkage=1e-3):
        self.shrinkage = shrinkage
        self.means_ = None
        self.prec_ = None

    def fit(self, feats, labels):
        feats = np.asarray(feats, float)
        labels = np.asarray(labels, int)
        classes = np.unique(labels)
        d = feats.shape[1]
        means, centered = [], []
        for c in classes:
            fc = feats[labels == c]
            mu = fc.mean(0)
            means.append(mu)
            centered.append(fc - mu)
        self.means_ = np.stack(means)
        # pooled within-class covariance: sum of class-centred outer products
        # divided by (N - K). Using np.cov here would re-centre the stacked
        # deviations by their global mean and divide by N-1, biasing the estimate
        # under class imbalance (Lee et al. 2018 use the pooled estimator).
        M = np.vstack(centered)
        C = (M.T @ M) / max(1, len(M) - len(classes))
        C = C + self.shrinkage * np.eye(d)
        self.prec_ = np.linalg.pinv(C)
        return self

    def score(self, feats):
        """Min Mahalanobis distance to any class mean (higher = more OOD)."""
        feats = np.asarray(feats, float)
        out = np.empty((len(feats), len(self.means_)))
        for j, mu in enumerate(self.means_):
            diff = feats - mu
            out[:, j] = np.einsum("ni,ij,nj->n", diff, self.prec_, diff)
        return out.min(1)


def auroc(score_in, score_ood):
    """AUROC for separating OOD (positive) from in-distribution via score."""
    s = np.concatenate([np.asarray(score_ood, float), np.asarray(score_in, float)])
    y = np.concatenate([np.ones(len(score_ood)), np.zeros(len(score_in))])
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def fpr_at_tpr(score_in, score_ood, tpr=0.95):
    """False-positive rate (ID flagged as OOD) at a target OOD true-positive rate."""
    score_in = np.asarray(score_in, float)
    score_ood = np.asarray(score_ood, float)
    thr = np.quantile(score_ood, 1 - tpr)  # threshold catching `tpr` of OOD
    return float(np.mean(score_in >= thr))
