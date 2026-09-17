"""Candidate models.

Each candidate is a scikit-learn :class:`~sklearn.pipeline.Pipeline` whose first
steps are the imputer and scaler. Bundling preprocessing into the pipeline is
what makes leakage-free walk-forward validation possible: ``fit`` on a training
fold fits the imputer/scaler on that fold only, and ``predict_proba`` on the
test fold applies the already-fitted transforms.
"""

from __future__ import annotations

from typing import Callable

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CANDIDATE_NAMES = ("logistic_regression", "random_forest", "gradient_boosting")


def build_pipeline(estimator, scale: bool = True) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _logistic(random_state: int) -> Pipeline:
    return build_pipeline(
        LogisticRegression(
            C=0.5,
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        scale=True,
    )


def _random_forest(random_state: int) -> Pipeline:
    return build_pipeline(
        RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=25,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        scale=False,
    )


def _gradient_boosting(random_state: int) -> Pipeline:
    return build_pipeline(
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            min_samples_leaf=25,
            random_state=random_state,
        ),
        scale=False,
    )


_BUILDERS: dict[str, Callable[[int], Pipeline]] = {
    "logistic_regression": _logistic,
    "random_forest": _random_forest,
    "gradient_boosting": _gradient_boosting,
}


def build_candidates(random_state: int = 7) -> dict[str, Pipeline]:
    """Fresh, unfitted candidate pipelines keyed by name."""
    return {name: builder(random_state) for name, builder in _BUILDERS.items()}


def build_candidate(name: str, random_state: int = 7) -> Pipeline:
    if name not in _BUILDERS:
        raise KeyError(f"Unknown model: {name}")
    return _BUILDERS[name](random_state)
