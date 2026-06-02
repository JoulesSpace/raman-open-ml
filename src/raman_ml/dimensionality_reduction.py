"""Dimensionality reduction and unsupervised structure for spectra.

Covers the common chemometric methods (and what BoxSERS exposes), with one honest distinction:

* **Linear, invertible, loadings-interpretable**: PCA. ``SpectroPCA`` is plain PCA
  (same maths as sklearn) plus the spectroscopy value-add BoxSERS provides -
  score scatter coloured by class and, crucially, **loadings plotted as spectra**
  so you can read which Raman bands define each component.
* **Non-linear manifold embeddings** (genuinely different algorithms, not
  wrappers): t-SNE, UMAP, MDS, Isomap - good for visualising cluster structure,
  but not invertible and not for downstream regression.
* **Supervised** projection: LDA (uses labels to maximise class separation).

``embedding_separability`` quantifies how well a 2-D embedding preserves class
structure (silhouette + a kNN cross-val accuracy in the embedding), so the DR
methods can be *benchmarked*, not just eyeballed.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class SpectroPCA:
    """PCA for spectra (same algorithm as sklearn PCA) with spectro plotting.

    The difference from "normal PCA" is not the maths - it is the interpretation
    helpers: ``loadings`` returns each component as a pseudo-spectrum over the
    wavenumber axis (peaks there = the bands that component encodes), and
    ``scatter``/``explained`` give the standard chemometric score/scree views.
    """

    def __init__(self, n_components=10, standardize=True):
        self.n_components = n_components
        self.standardize = standardize
        self.scaler_ = StandardScaler() if standardize else None
        self.pca_ = PCA(n_components=n_components, random_state=0)

    def fit(self, X):
        Z = self.scaler_.fit_transform(X) if self.standardize else np.asarray(X)
        self.pca_.fit(Z)
        return self

    def transform(self, X):
        Z = self.scaler_.transform(X) if self.standardize else np.asarray(X)
        return self.pca_.transform(Z)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @property
    def explained_variance_ratio_(self):
        return self.pca_.explained_variance_ratio_

    def loadings(self):
        """Component loadings as (n_components, n_wavenumbers) pseudo-spectra."""
        return self.pca_.components_


def embed_2d(X, method="pca", y=None, seed=0, standardize=True, **kwargs):
    """Project spectra to 2-D with the chosen method.

    method in {"pca", "tsne", "umap", "mds", "isomap", "lda"}. "lda" is
    supervised and needs ``y``. Returns an (n, 2) array.
    """
    Z = StandardScaler().fit_transform(X) if standardize else np.asarray(X)
    method = method.lower()
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(Z)
    if method == "lda":
        if y is None:
            raise ValueError("LDA is supervised; pass y.")
        n_comp = min(2, len(np.unique(y)) - 1)
        emb = LinearDiscriminantAnalysis(n_components=n_comp).fit_transform(Z, y)
        if emb.shape[1] == 1:  # binary -> pad to 2-D for plotting
            emb = np.column_stack([emb[:, 0], np.zeros(len(emb))])
        return emb
    if method == "tsne":
        perplexity = kwargs.get("perplexity", min(30, max(5, len(Z) // 4)))
        return TSNE(n_components=2, perplexity=perplexity, init="pca",
                    random_state=seed).fit_transform(Z)
    if method == "umap":
        import umap  # optional dependency
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(Z)
    if method == "mds":
        return MDS(n_components=2, random_state=seed,
                   normalized_stress="auto").fit_transform(Z)
    if method == "isomap":
        n_neighbors = kwargs.get("n_neighbors", min(10, len(Z) - 1))
        return Isomap(n_components=2, n_neighbors=n_neighbors).fit_transform(Z)
    raise ValueError(f"unknown method: {method!r}")


def embedding_separability(emb, y, cv=5, seed=0):
    """How well a *fixed* 2-D embedding preserves class structure.

    Returns silhouette (cluster geometry) and kNN cross-val accuracy (can a
    simple classifier recover the labels from the 2-D coords).

    IMPORTANT - this measures *cluster recoverability of an already-computed
    embedding*, not out-of-sample generalisation: t-SNE/UMAP/Isomap are
    transductive (each point's coords used all other points) and **LDA is
    supervised** (it used ``y`` to build the projection), so LDA's kNN-accuracy
    is optimistic and not comparable to the unsupervised methods. Use it to rank
    the unsupervised embeddings against each other, and read LDA as an upper
    bound, not a fair peer.
    """
    emb = np.asarray(emb)
    y = np.asarray(y)
    sil = float(silhouette_score(emb, y)) if len(np.unique(y)) > 1 else float("nan")
    folds = min(cv, np.bincount(y).min()) if y.dtype.kind in "iu" else cv
    folds = max(2, folds)
    knn_acc = float(cross_val_score(KNeighborsClassifier(5), emb, y,
                                    cv=folds).mean())
    return {"silhouette": sil, "knn_accuracy": knn_acc}
