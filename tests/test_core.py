"""Fast unit tests for raman_ml (no dataset downloads, no GPU, < a few seconds).

Run: pytest -q   (from the repo root)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from raman_ml import augment, ood
from raman_ml import calibration_transfer as ct
from raman_ml import preprocessing as P
from raman_ml import uncertainty as U
from raman_ml import variable_selection as VS


def _synthetic_spectra(n=60, L=200, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, L)
    peaks = np.exp(-((x - 0.3) ** 2) / 0.001) + 0.5 * np.exp(-((x - 0.7) ** 2) / 0.002)
    amp = rng.uniform(0.5, 1.5, (n, 1))
    base = (rng.uniform(0, 0.3, (n, 1)) * x[None, :])  # sloped baseline
    X = amp * peaks[None, :] + base + rng.normal(0, 0.01, (n, L))
    return X


# --- preprocessing ----------------------------------------------------------
@pytest.mark.parametrize("method", ["als", "arpls", "airpls"])
def test_baseline_reduces_offset(method):
    X = _synthetic_spectra()
    out = P.remove_baseline(X, method=method)
    # subtracting a (non-negative) baseline lowers the overall signal level
    assert out.shape == X.shape
    assert out.mean() < X.mean()


def test_norms_and_derivative_shapes():
    X = _synthetic_spectra()
    assert np.allclose(np.linalg.norm(P.l2_normalize(X), axis=1), 1.0, atol=1e-6)
    assert np.allclose(P.snv(X).mean(1), 0.0, atol=1e-6)
    assert P.savgol_derivative(X, window=11, poly=2, deriv=1).shape == X.shape
    assert P.msc(X)[0].shape == X.shape


def test_resample_handles_descending_axis():
    shift = np.linspace(1800, 400, 50)          # descending
    inten = np.linspace(0, 1, 50)
    grid = np.linspace(500, 1700, 30)
    out = P.resample_to_grid(shift, inten, grid)
    assert out.shape == (30,) and np.all(np.isfinite(out))


# --- conformal prediction ---------------------------------------------------
def test_conformal_regression_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 1000); pred = y + rng.normal(0, 0.3, 1000)
    lo, hi = U.conformal_interval(y[:500], pred[:500], pred[500:], alpha=0.1)
    cov = U.interval_metrics(y[500:], lo, hi)["coverage"]
    assert cov >= 0.85  # ~0.90 target, allow finite-sample slack


def test_conformal_classification_coverage_and_nonempty():
    rng = np.random.default_rng(0)
    K = 6
    logits = rng.normal(0, 1.5, (1000, K))
    labels = logits.argmax(1)  # learnable structure
    logits += rng.normal(0, 1.0, (1000, K))
    probs = U.softmax(logits)
    fn, _ = U.calibrate_conformal_classifier(probs[:500], labels[:500], alpha=0.1)
    masks = fn(probs[500:])
    sm = U.set_metrics(masks, labels[500:])
    assert sm["coverage"] >= 0.85
    assert masks.sum(1).min() >= 1  # never empty


def test_temperature_scaling_reduces_ece():
    rng = np.random.default_rng(0)
    K = 5
    logits = rng.normal(0, 1, (800, K)) * 4.0  # overconfident
    labels = rng.integers(0, K, 800)
    T = U.fit_temperature(logits[:400], labels[:400])
    ece_before = U.expected_calibration_error(U.softmax(logits[400:]), labels[400:])
    ece_after = U.expected_calibration_error(U.softmax(logits[400:], T), labels[400:])
    assert T > 1.0                       # should soften overconfident logits
    assert ece_after <= ece_before + 1e-6


def test_jackknife_plus_runs():
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 5)); y = X @ rng.normal(0, 1, 5) + rng.normal(0, 0.1, 40)
    lo, hi = U.jackknife_plus_interval(lambda: LinearRegression(), X, y, X[:10])
    assert lo.shape == (10,) and np.all(hi >= lo)


# --- OOD --------------------------------------------------------------------
def test_ood_detectors_separate_clear_ood():
    rng = np.random.default_rng(0)
    K = 5
    li = rng.normal(0, 1, (300, K)); li[np.arange(300), rng.integers(0, K, 300)] += 6
    lo = rng.normal(0, 0.2, (300, K))  # flat -> OOD
    assert ood.auroc(ood.energy_score(li), ood.energy_score(lo)) > 0.9
    feats_in = rng.normal(0, 1, (300, 8)); lab = rng.integers(0, 4, 300)
    m = ood.MahalanobisOOD().fit(feats_in, lab)
    feats_ood = rng.normal(8, 1, (200, 8))
    assert ood.auroc(m.score(feats_in), m.score(feats_ood)) > 0.9
    assert 0.0 <= ood.fpr_at_tpr(m.score(feats_in), m.score(feats_ood)) <= 1.0


# --- calibration transfer ---------------------------------------------------
def test_pds_reduces_transfer_error():
    rng = np.random.default_rng(0)
    prim = np.abs(rng.normal(0, 1, (30, 80)))
    sec = prim * 1.4 + 0.3 + rng.normal(0, 0.01, (30, 80))  # gain+offset shift
    pds = ct.PiecewiseDirectStandardization(half_window=4).fit(sec[:15], prim[:15])
    mapped = pds.transform(sec[15:])
    err_before = np.mean(np.abs(sec[15:] - prim[15:]))
    err_after = np.mean(np.abs(mapped - prim[15:]))
    assert err_after < err_before


def test_apply_transfer_if_improves_guard():
    # exercise the guard's decision logic with stub transforms (independent of
    # PDS reconstruction quality)
    from sklearn.linear_model import LinearRegression

    class _Stub:
        def __init__(self, fn): self.fn = fn
        def transform(self, X): return self.fn(np.asarray(X, float))

    rng = np.random.default_rng(0)
    prim = rng.normal(0, 1, (40, 6))
    yv = prim[:, 1] * 2 + rng.normal(0, 0.05, 40)
    model = LinearRegression().fit(prim[:20], yv[:20])
    sec = prim * 1.5 + 0.4                                    # affine shift
    good = _Stub(lambda X: (X - 0.4) / 1.5)                   # recovers primary
    _, used = ct.apply_transfer_if_improves(model, good, sec[20:30], yv[20:30],
                                            sec[30:])
    assert used                                              # transfer clearly helps
    bad = _Stub(lambda X: X + rng.normal(0, 5, X.shape))     # only adds noise
    out, used2 = ct.apply_transfer_if_improves(model, bad, prim[20:30], yv[20:30],
                                               prim[30:])
    assert not used2 and np.allclose(out, prim[30:])         # falls back to raw


def test_select_transfer_standards_unique():
    X = np.random.default_rng(0).normal(0, 1, (50, 20))
    idx = ct.select_transfer_standards(X, n=10)
    assert len(idx) == len(set(idx.tolist())) and len(idx) <= 10


# --- augmentation -----------------------------------------------------------
def test_augment_preserves_shape_and_changes_values():
    X = _synthetic_spectra()
    aug = augment.SpectralAugment(seed=1, p=1.0)
    out = aug(X)
    assert out.shape == X.shape
    assert not np.allclose(out, X)


def test_mixup_soft_labels_sum_to_one():
    X = _synthetic_spectra(n=20)
    y = np.random.default_rng(0).integers(0, 4, 20)
    Xm, Ym = augment.mixup(X, y, alpha=0.4, n_classes=4,
                           rng=np.random.default_rng(0))
    assert Xm.shape == X.shape
    assert np.allclose(Ym.sum(1), 1.0)


# --- deep ensemble + interpretability (light CNN training) ------------------
def _two_class_data(n=120, L=120, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, L)
    y = rng.integers(0, 2, n)
    centre = np.where(y == 0, 0.3, 0.6)
    X = np.array([np.exp(-((x - c) ** 2) / 0.002) for c in centre])
    X += rng.normal(0, 0.02, (n, L))
    return X.astype(np.float32), y


def test_deep_ensemble_classifier_runs():
    from raman_ml.models import CNNClassifier, DeepEnsemble
    X, y = _two_class_data()
    ens = DeepEnsemble(lambda: CNNClassifier(n_out=2, epochs=3, batch_size=32),
                       n_members=3).fit(X, y)
    proba = ens.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(1), 1.0, atol=1e-5)


def test_weighted_ensemble_regressor():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    from raman_ml.models import WeightedEnsembleRegressor
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (120, 6))
    y = X[:, 0] * 2 - X[:, 2] + rng.normal(0, 0.1, 120)
    ens = WeightedEnsembleRegressor(
        factories=[lambda: LinearRegression(),
                   lambda: RandomForestRegressor(60, random_state=0)],
        weights="cv", cv=4).fit(X[:90], y[:90])
    pred = ens.predict(X[90:])
    assert pred.shape == (30,)
    assert abs(ens.weights_.sum() - 1.0) < 1e-9
    assert r2_score(y[90:], pred) > 0.8     # ensemble tracks a learnable target


def test_vip_scores_and_selection():
    from sklearn.cross_decomposition import PLSRegression
    rng = np.random.default_rng(0)
    L = 60
    X = rng.normal(0, 1, (80, L))
    informative = [10, 11, 12, 40]
    y = X[:, informative].sum(1) + rng.normal(0, 0.1, 80)
    pls = PLSRegression(n_components=3).fit(X, y)
    vip = VS.vip_scores(pls)
    assert vip.shape == (L,)
    # informative variables should rank among the highest VIP
    top = set(np.argsort(-vip)[:8].tolist())
    assert len(top & set(informative)) >= 2
    assert len(VS.select_by_vip(pls, top_k=10)) == 10


def test_tuning_grid_random_improve_or_match():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    from raman_ml.tuning import tune
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (120, 8))
    y = X[:, 0] * 2 - X[:, 3] + rng.normal(0, 0.1, 120)
    space = {"n_estimators": [50, 150], "max_depth": [2, 4, 8],
             "max_features": [0.3, 0.6, 1.0]}
    for method in ("grid", "random"):
        best, params, score = tune(RandomForestRegressor(random_state=0), space,
                                   X, y, method=method, scoring="r2", cv=4,
                                   n_iter=6, random_state=0)
        assert set(params).issubset(space)
        assert -1.0 <= score <= 1.0
        # tuned should not be worse than an arbitrary default config
        default = cross_val_score(RandomForestRegressor(n_estimators=50,
                                  max_depth=2, random_state=0), X, y, cv=4,
                                  scoring="r2").mean()
        assert score >= default - 1e-6


def test_tuning_bayes_optional():
    pytest.importorskip("optuna")
    from sklearn.svm import SVR

    from raman_ml.tuning import tune
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (80, 5))
    y = X[:, 1] * 1.5 + rng.normal(0, 0.1, 80)
    space = {"C": (0.1, 100.0), "gamma": (1e-3, 1.0)}
    best, params, score = tune(SVR(), space, X, y, method="bayes", scoring="r2",
                               cv=4, n_iter=12, random_state=0)
    assert "C" in params and "gamma" in params
    assert score <= 1.0


def test_shap_wavenumber_importance_optional():
    pytest.importorskip("shap")
    from sklearn.ensemble import RandomForestRegressor

    from raman_ml.interpretability import shap_wavenumber_importance
    rng = np.random.default_rng(0)
    L = 40
    X = rng.normal(0, 1, (80, L))
    informative = [5, 6, 25]
    y = X[:, informative].sum(1) + rng.normal(0, 0.05, 80)
    rf = RandomForestRegressor(n_estimators=80, random_state=0).fit(X, y)
    imp = shap_wavenumber_importance(rf, X, X[:30])
    assert imp.shape == (L,)
    # informative wavenumbers should be among the most important
    assert len(set(np.argsort(-imp)[:6]) & set(informative)) >= 2


@pytest.mark.parametrize("method", ["modpoly", "imodpoly", "snip", "rubberband"])
def test_extra_baselines_run(method):
    X = _synthetic_spectra()
    out = P.remove_baseline(X, method=method)
    assert out.shape == X.shape and np.all(np.isfinite(out))


def test_snip_inverse_roundtrip_and_baseline_below():
    # the SNIP log-log-sqrt transform must invert exactly, and the baseline
    # must sit at/under the signal
    rng = np.random.default_rng(0)
    y = np.abs(rng.normal(5, 1, 200)) + np.linspace(0, 3, 200)
    offset = y.min(); t = y - offset + 1
    v = np.log(np.log(np.sqrt(t) + 1) + 1)
    inv = (np.exp(np.exp(v) - 1) - 1) ** 2 - 1 + offset
    assert np.max(np.abs(inv - y)) < 1e-8
    out = P.remove_baseline(y[None, :], method="snip")
    assert np.all(out[0] <= y + 1e-6)


def test_rubberband_is_lower_hull():
    x = np.linspace(0, 1, 300)
    peaks = np.exp(-((x - 0.4) ** 2) / 0.002)
    y = 0.5 + 0.8 * x + peaks                       # rising baseline + peak
    b = P.rubberband_baseline(y)
    assert np.all(b <= y + 1e-9)                    # lower hull, not upper
    assert (y - b).max() > 0.8                      # peak preserved after removal


def test_jackknife_plus_coverage_small_n():
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(1)
    covs = []
    for _ in range(15):
        X = rng.normal(0, 1, (48, 5)); w = rng.normal(0, 1, 5)
        yv = X @ w + rng.normal(0, 0.3, 48)
        lo, hi = U.jackknife_plus_interval(lambda: LinearRegression(),
                                           X[:36], yv[:36], X[36:], alpha=0.1)
        covs.append(np.mean((yv[36:] >= lo) & (yv[36:] <= hi)))
    assert np.mean(covs) >= 0.85          # finite-sample correction -> ~0.90


def test_cosmic_ray_removal():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 300)
    base = np.exp(-((x - 0.5) ** 2) / 0.01)            # smooth peak
    X = np.tile(base, (4, 1)) + rng.normal(0, 0.005, (4, 300))
    X[1, 150] += 5.0                                    # inject a cosmic spike
    out = P.remove_cosmic_rays(X, threshold=6.0)
    assert out.shape == X.shape
    assert out[1, 150] < X[1, 150] - 1.0                # spike attenuated
    assert abs(out[0, 75] - X[0, 75]) < 1e-6            # clean rows untouched


def test_peak_feature_extractor():
    from raman_ml.peaks import PeakFeatureExtractor, area_ratio
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 200)
    # two bands; amplitude of band A encodes the target
    amp = rng.uniform(0.5, 2.0, 60)
    A = amp[:, None] * np.exp(-((x - 0.3) ** 2) / 0.001)
    B = np.exp(-((x - 0.7) ** 2) / 0.001)
    X = A + B + rng.normal(0, 0.01, (60, 200))
    pf = PeakFeatureExtractor(n_bands=5, window=6).fit(X)
    F = pf.transform(X)
    assert F.shape[0] == 60
    assert F.shape[1] == len(pf.bands_) * 4 == len(pf.feature_names_)
    assert 2 <= len(pf.bands_) <= 5            # finds the (>=2) real bands
    r = area_ratio(X[0], np.arange(200), (50, 70), (130, 150))
    assert np.isfinite(r)


def test_ssl_mae_pretrain_finetune_runs():
    from raman_ml.ssl import MAEClassifier
    rng = np.random.default_rng(0)
    L = 200
    x = np.linspace(0, 1, L)
    y = rng.integers(0, 3, 150)
    centre = np.array([0.3, 0.5, 0.7])[y]
    X = np.array([np.exp(-((x - c) ** 2) / 0.002) for c in centre]).astype(np.float32)
    X += rng.normal(0, 0.02, (150, L)).astype(np.float32)
    clf = MAEClassifier(n_out=3, base=16, pretrain_epochs=3, finetune_epochs=3,
                        batch_size=32, patch=20)
    clf.pretrain(X).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (150, 3)
    assert np.allclose(proba.sum(1), 1.0, atol=1e-5)


def test_cnn_save_load_roundtrip(tmp_path):
    from raman_ml.models import CNNClassifier, load_cnn
    X, y = _two_class_data(L=128)
    clf = CNNClassifier(n_out=2, epochs=3, batch_size=32).fit(X, y)
    p = clf.predict_proba(X)
    path = str(tmp_path / "m.pt")
    clf.save(path)
    loaded = load_cnn(path)
    pl = loaded.predict_proba(X)
    # identical class predictions; probs match up to GPU-vs-CPU float diffs
    assert np.array_equal(pl.argmax(1), p.argmax(1))
    assert np.allclose(pl, p, atol=1e-3)


def test_set_global_determinism_runs():
    from raman_ml.models import set_global_determinism
    set_global_determinism(0)  # must not raise on CPU or GPU


def test_shap_per_class_importance_optional():
    pytest.importorskip("shap")
    from sklearn.ensemble import RandomForestClassifier

    from raman_ml.interpretability import shap_per_class_importance
    rng = np.random.default_rng(0)
    L = 40
    y = rng.integers(0, 3, 120)
    # each class has its own discriminative band
    band = {0: 8, 1: 20, 2: 32}
    X = rng.normal(0, 0.2, (120, L))
    for i, c in enumerate(y):
        X[i, band[c]] += 3.0
    rf = RandomForestClassifier(n_estimators=120, random_state=0).fit(X, y)
    imp = shap_per_class_importance(rf, X, X, y)
    assert imp.shape == (3, L)
    # each class's own band should be among its most important wavenumbers
    for c in range(3):
        assert band[c] in set(np.argsort(-imp[c])[:5])


def test_gradcam_1d_runs():
    from raman_ml.interpretability import grad_cam_1d
    from raman_ml.models import CNNClassifier
    X, y = _two_class_data(L=128)
    clf = CNNClassifier(n_out=2, epochs=3, batch_size=32).fit(X, y)
    cam = grad_cam_1d(clf, X[:5])
    assert cam.shape == (5, 128)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6


def test_dimreduction_pca_and_separability():
    from raman_ml.dimensionality_reduction import SpectroPCA, embed_2d, embedding_separability
    rng = np.random.default_rng(0)
    # two separated blobs in spectral space
    L = 50
    y = rng.integers(0, 2, 120)
    X = rng.normal(0, 0.2, (120, L))
    X[y == 1, 10:20] += 3.0
    sp = SpectroPCA(n_components=5).fit(X)
    assert sp.transform(X).shape == (120, 5)
    assert sp.loadings().shape == (5, L)
    for mth in ("pca", "lda", "tsne"):
        emb = embed_2d(X, method=mth, y=y, seed=0)
        assert emb.shape == (120, 2)
    sep = embedding_separability(embed_2d(X, "pca"), y)
    assert sep["knn_accuracy"] > 0.8  # separable blobs recoverable in 2-D


def test_msresnet_multiscale_runs():
    import torch

    from raman_ml.models import CNNClassifier, MSResNet1D
    m = MSResNet1D(200, 4, base=16, layers=(2, 2), se=True)
    assert m(torch.randn(3, 1, 200)).shape == (3, 4)
    assert m.embed(torch.randn(3, 1, 200)).shape == (3, 16 * 2)
    X, y = _two_class_data(L=160)
    clf = CNNClassifier(n_out=2, epochs=2, batch_size=32, arch="msresnet",
                        resnet_base=16, resnet_layers=(2, 2), se=True)
    clf.fit(X, y)
    assert clf.predict_proba(X).shape == (len(X), 2)


def test_finetune_and_se_resnet_run():
    from raman_ml.models import CNNClassifier
    Xs, ys = _two_class_data(seed=1)
    Xt, yt = _two_class_data(seed=2)
    clf = CNNClassifier(n_out=2, epochs=3, batch_size=32, arch="resnet",
                        resnet_base=16, se=True)
    clf.fit(Xs, ys)                       # pretrain
    clf.finetune(Xt, yt, epochs=2)        # warm-start adapt
    assert clf.predict(Xt).shape == (len(Xt),)
    assert clf.predict_proba(Xt).shape == (len(Xt), 2)


def test_integrated_gradients_shape_and_completeness():
    import torch

    from raman_ml.interpretability import integrated_gradients
    from raman_ml.models import CNNRegressor
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 80)).astype(np.float32)
    y = X[:, 20] * 2.0 + rng.normal(0, 0.05, 40)  # depends on one band
    reg = CNNRegressor(epochs=4, batch_size=16).fit(X, y)
    attr = integrated_gradients(reg, X[:5], steps=32)
    assert attr.shape == (5, 80)
    # completeness: sum(attr) ~ f(x) - f(baseline)
    base = np.zeros((1, 80), np.float32)
    with torch.no_grad():
        fx = reg._raw_predict(X[:5]).ravel()
        fb = reg._raw_predict(base).ravel()[0]
    assert np.allclose(attr.sum(1), fx - fb, atol=0.05 * (abs(fx).mean() + 1e-6) + 0.05)
