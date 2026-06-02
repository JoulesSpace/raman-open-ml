"""Peak attribution for spectral neural nets via Integrated Gradients.

Interpretability (which wavenumbers drove a prediction) is repeatedly named as a
priority gap in Raman-ML reviews. Integrated Gradients (Sundararajan et al. 2017)
attributes a model output to each input wavenumber by integrating gradients along
a straight path from a baseline (zeros) to the input, with completeness:
sum(attributions) == f(input) - f(baseline).
"""
from __future__ import annotations

import numpy as np
import torch


def integrated_gradients(cnn, X, target=None, baseline=None, steps=64):
    """Integrated Gradients attributions for a fitted CNN wrapper.

    Parameters
    ----------
    cnn : a fitted CNNClassifier / CNNRegressor (has .model and .device)
    X   : (n, L) spectra
    target : class index per row for classifiers (defaults to argmax);
             ignored for single-output regressors.
    baseline : (L,) baseline spectrum (defaults to zeros).
    steps : Riemann steps along the path.

    Returns attributions of shape (n, L); larger magnitude = more influential
    wavenumber.
    """
    model = cnn.model
    device = cnn.device
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    n, L = X.shape
    base = np.zeros(L, np.float32) if baseline is None else np.asarray(baseline, np.float32)
    Xt = torch.tensor(X, device=device)
    bt = torch.tensor(base, device=device)[None, :]

    is_clf = model(torch.zeros(1, 1, L, device=device)).shape[1] > 1
    if is_clf and target is None:
        with torch.no_grad():
            target = model(Xt.unsqueeze(1)).argmax(1).cpu().numpy()

    total = torch.zeros(n, L, device=device)
    for s in range(1, steps + 1):
        a = s / steps
        x = (bt + a * (Xt - bt)).unsqueeze(1).clone().requires_grad_(True)
        out = model(x)
        if is_clf:
            tgt = torch.tensor(target, device=device, dtype=torch.long)
            sel = out.gather(1, tgt[:, None]).sum()
        else:
            sel = out.sum()
        grad = torch.autograd.grad(sel, x)[0].squeeze(1)
        total = total + grad
    attributions = (Xt - bt) * total / steps
    return attributions.detach().cpu().numpy()


def grad_cam_1d(cnn, X, target=None, conv_module=None):
    """Grad-CAM (Selvaraju et al. 2017) for a 1-D spectral CNN.

    Uses the gradient of the target output w.r.t. the last conv feature maps,
    GAP-weighted and ReLU'd, then upsampled to the input wavenumber axis - a
    class-discriminative localisation map (which spectral region drove the
    class). Complements Integrated Gradients (per-input attribution).

    Returns a (n, L) array in [0, 1] per sample.
    """
    import torch.nn as nn
    import torch.nn.functional as F
    model, device = cnn.model, cnn.device
    model.eval()
    if conv_module is None:
        convs = [m for m in model.modules() if isinstance(m, nn.Conv1d)]
        conv_module = convs[-1]
    store = {}
    h1 = conv_module.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    h2 = conv_module.register_full_backward_hook(
        lambda m, gi, go: store.__setitem__("g", go[0]))
    try:
        Xt = torch.tensor(np.asarray(X, np.float32)).unsqueeze(1).to(device)
        out = model(Xt)
        if out.shape[1] > 1:  # classifier
            tgt = (out.argmax(1) if target is None
                   else torch.tensor(np.atleast_1d(target), device=device,
                                     dtype=torch.long))
            score = out.gather(1, tgt.view(-1, 1)).sum()
        else:
            score = out.sum()
        model.zero_grad()
        score.backward()
        A, G = store["a"], store["g"]              # (B, C, L')
        w = G.mean(dim=2, keepdim=True)            # GAP of gradients
        cam = torch.relu((w * A).sum(1))           # (B, L')
        cam = F.interpolate(cam.unsqueeze(1), size=np.asarray(X).shape[1],
                            mode="linear", align_corners=False).squeeze(1)
        cam = cam.detach().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()
    cam = cam - cam.min(1, keepdims=True)
    mx = cam.max(1, keepdims=True)
    mx[mx == 0] = 1.0
    return cam / mx


def top_peaks(attributions, wavenumbers, k=10):
    """Return the k most influential wavenumbers (by mean |attribution|)."""
    imp = np.abs(np.atleast_2d(attributions)).mean(0)
    idx = np.argsort(-imp)[:k]
    wn = np.asarray(wavenumbers)
    return [(float(wn[i]), float(imp[i])) for i in sorted(idx, key=lambda j: -imp[j])]


# --------------------------------------------------------------------------- #
# SHAP (model-agnostic peak attribution for sklearn models)                    #
# --------------------------------------------------------------------------- #
def shap_explainer(model, background):
    """Pick the right SHAP explainer for a fitted sklearn model.

    Tree models -> exact TreeExplainer; linear models -> LinearExplainer;
    anything else -> model-agnostic KernelExplainer (slow on 1000 wavenumbers).
    """
    import shap
    name = model.__class__.__name__.lower()
    is_tree = any(x in name for x in ("forest", "tree", "xgb", "lgbm",
                                      "catboost", "boosting", "gbm"))
    try:
        if is_tree:
            return shap.TreeExplainer(model)
        if hasattr(model, "coef_"):
            return shap.LinearExplainer(model, background)
        return shap.KernelExplainer(model.predict, background)
    except Exception:  # pragma: no cover - fall back to the unified API
        return shap.Explainer(model, background)


def shap_wavenumber_importance(model, X_background, X_explain,
                               max_background=100, seed=0):
    """Mean |SHAP value| per wavenumber for a fitted sklearn model.

    Returns a length-L importance vector (averaged over explained samples and,
    for classifiers, over classes) - i.e. which wavenumbers drive predictions.
    """
    import shap  # noqa: F401  (import error surfaces clearly if shap is missing)
    rng = np.random.default_rng(seed)
    Xb = np.asarray(X_background, float)
    if len(Xb) > max_background:
        Xb = Xb[rng.choice(len(Xb), max_background, replace=False)]
    expl = shap_explainer(model, Xb)
    try:
        vals = np.asarray(expl(np.asarray(X_explain, float)).values)
    except Exception:  # pragma: no cover - older explainers
        vals = np.asarray(expl.shap_values(np.asarray(X_explain, float)))
    if vals.ndim == 3:
        # (n, L, C) or (C, n, L) -> reduce over everything but the L axis
        l_axis = 1 if vals.shape[1] == np.asarray(X_explain).shape[1] else 2
        axes = tuple(a for a in range(3) if a != l_axis)
        return np.abs(vals).mean(axis=axes)
    return np.abs(vals).mean(axis=0)


def shap_per_class_importance(model, X_background, X_explain, y_explain=None,
                              max_background=100, seed=0):
    """Per-class mean |SHAP| per wavenumber for a multiclass model.

    Returns (n_classes, n_wavenumbers): for class c, the average |SHAP value for
    class c| over the explained samples whose true label is c (or over all
    samples if labels are not given) - i.e. which bands drive *that* substance's
    prediction. Feed this to a heatmap to see the diagnostic bands per class.
    """
    import shap  # noqa: F401
    rng = np.random.default_rng(seed)
    Xb = np.asarray(X_background, float)
    if len(Xb) > max_background:
        Xb = Xb[rng.choice(len(Xb), max_background, replace=False)]
    Xe = np.asarray(X_explain, float)
    vals = np.asarray(shap_explainer(model, Xb)(Xe).values)
    if vals.ndim != 3:
        raise ValueError("expected multiclass SHAP values of shape (n, L, C)")
    n_classes = len(getattr(model, "classes_", range(vals.shape[-1])))
    L = Xe.shape[1]
    # normalise layout to (n, L, C)
    if vals.shape[1] != L and vals.shape[2] == L:        # (n, C, L) or (C, n, L)
        vals = np.moveaxis(vals, -1, 1)
    out = np.zeros((n_classes, L))
    y = None if y_explain is None else np.asarray(y_explain)
    for c in range(n_classes):
        m = (y == c) if (y is not None and np.any(y == c)) else slice(None)
        out[c] = np.abs(vals[m, :, c]).mean(axis=0)
    return out
