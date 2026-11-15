"""Train-only preprocessing contract.

The dataset stores raw feature values, including possible missing sensor
readings. Imputation and scaling are fitted on the training split only and
then applied unchanged to validation, test and live hybrid inference.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor(scale: bool = False) -> Pipeline:
    """Return a fresh median-imputation pipeline.

    ``SimpleImputer(add_indicator=True)`` creates missingness flags only for
    columns that contain missing values in the training data. Those flags are
    part of the fitted preprocessing state, so validation/test never influence
    them.
    """
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def fit_preprocessor(
    X_train: pd.DataFrame,
    scale: bool = False,
) -> Pipeline:
    """Fit preprocessing on training data only."""
    if len(X_train) == 0:
        raise ValueError("X_train must contain at least one row")
    return build_preprocessor(scale=scale).fit(X_train)


def transform(
    preprocessor: Pipeline,
    X: pd.DataFrame,
) -> Any:
    """Apply a previously fitted preprocessor."""
    return preprocessor.transform(X)


def build_model_pipeline(
    estimator: Any,
    scale: bool = False,
) -> Pipeline:
    """Combine the shared preprocessor with one model estimator."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(scale=scale)),
            ("model", estimator),
        ]
    )
