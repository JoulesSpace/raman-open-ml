"""Hyperparameter tuning for the classical models: grid / random / Bayesian.

Adapted (streamlined) from the AutoML platform ml-training tuning module. A single
``tune`` entry point dispatches to:

* ``grid``   - exhaustive GridSearchCV,
* ``random`` - RandomizedSearchCV (good default for a few continuous params),
* ``bayes``  - Optuna TPE Bayesian search (sample-efficient; optional dependency).

Works on any scikit-learn estimator or Pipeline. ``param_space`` maps a parameter
name (use ``step__param`` for Pipeline steps) to either a list of candidate values
(grid / categorical) or a ``(low, high)`` tuple of bounds (Bayesian; log-scaled
automatically when the range spans >= 2 orders of magnitude).
"""
from __future__ import annotations

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score


def _as_lists(space):
    out = {}
    for k, v in space.items():
        if isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = [v]
    return out


def _suggest(trial, key, value):
    # (low, high) -> numeric search; list -> categorical
    if isinstance(value, tuple) and len(value) == 2:
        low, high = value
        if isinstance(low, int) and isinstance(high, int):
            return trial.suggest_int(key, low, high)
        log = low > 0 and high / low >= 100
        return trial.suggest_float(key, low, high, log=log)
    return trial.suggest_categorical(key, list(value))


def tune(estimator, param_space, X, y, *, method="random", scoring="r2",
         cv=5, n_iter=30, n_jobs=-1, random_state=0):
    """Return ``(best_estimator, best_params, best_score)``.

    ``best_estimator`` is refit on all of (X, y). ``best_score`` is the mean
    cross-validated ``scoring`` of the best configuration.
    """
    if method == "grid":
        s = GridSearchCV(clone(estimator), _as_lists(param_space),
                         scoring=scoring, cv=cv, n_jobs=n_jobs)
        s.fit(X, y)
        return s.best_estimator_, s.best_params_, float(s.best_score_)

    if method == "random":
        s = RandomizedSearchCV(clone(estimator), _as_lists(param_space),
                               n_iter=n_iter, scoring=scoring, cv=cv,
                               n_jobs=n_jobs, random_state=random_state)
        s.fit(X, y)
        return s.best_estimator_, s.best_params_, float(s.best_score_)

    if method == "bayes":
        try:
            import optuna
        except ImportError as e:  # pragma: no cover
            raise ImportError("method='bayes' needs optuna (pip install optuna)") from e

        def objective(trial):
            params = {k: _suggest(trial, k, v) for k, v in param_space.items()}
            model = clone(estimator).set_params(**params)
            return cross_val_score(model, X, y, cv=cv, scoring=scoring,
                                   n_jobs=n_jobs).mean()

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_iter)
        best = clone(estimator).set_params(**study.best_params).fit(X, y)
        return best, study.best_params, float(study.best_value)

    raise ValueError(f"unknown tuning method: {method!r}")
