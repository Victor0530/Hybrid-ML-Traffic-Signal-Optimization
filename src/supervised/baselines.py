"""Naive baselines used to judge whether ML adds value.

The persistence forecast predicts that the future count equals the latest
observed count. In every feature set the first column is ``vehicle_count``, so
the estimator can also be used after the shared preprocessing pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class PersistenceRegressor(RegressorMixin, BaseEstimator):
    """Predict ``vehicle_count`` at the prediction time.

    Subclasses ``BaseEstimator``/``RegressorMixin`` so the estimator satisfies
    the scikit-learn estimator protocol: without them ``Pipeline.predict`` fails
    its ``check_is_fitted`` introspection under scikit-learn >= 1.6. The fitted
    attribute must end in a trailing underscore for that check to succeed.
    """

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "PersistenceRegressor":
        if isinstance(X, pd.DataFrame):
            if "vehicle_count" not in X.columns:
                raise ValueError("PersistenceRegressor needs a vehicle_count column")
            self.column_index_ = int(X.columns.get_loc("vehicle_count"))
        else:
            array = np.asarray(X)
            if array.ndim != 2 or array.shape[1] == 0:
                raise ValueError("PersistenceRegressor needs a 2-D feature matrix")
            self.column_index_ = 0
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not hasattr(self, "column_index_"):
            raise RuntimeError("PersistenceRegressor must be fitted first")
        if isinstance(X, pd.DataFrame):
            if "vehicle_count" in X.columns:
                values = X["vehicle_count"]
            else:
                values = X.iloc[:, self.column_index_]
        else:
            values = np.asarray(X)[:, self.column_index_]
        return np.asarray(values, dtype=float)
