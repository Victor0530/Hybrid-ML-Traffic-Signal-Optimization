"""Naive baselines used to judge whether ML adds value.

The persistence forecast predicts that the future count equals the latest
observed count. In every feature set the first column is ``vehicle_count``, so
the estimator can also be used after the shared preprocessing pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class PersistenceRegressor:
    """Predict ``vehicle_count`` at the prediction time."""

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "PersistenceRegressor":
        if isinstance(X, pd.DataFrame):
            if "vehicle_count" not in X.columns:
                raise ValueError("PersistenceRegressor needs a vehicle_count column")
            self._column_index = int(X.columns.get_loc("vehicle_count"))
        else:
            array = np.asarray(X)
            if array.ndim != 2 or array.shape[1] == 0:
                raise ValueError("PersistenceRegressor needs a 2-D feature matrix")
            self._column_index = 0
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not hasattr(self, "_column_index"):
            raise RuntimeError("PersistenceRegressor must be fitted first")
        if isinstance(X, pd.DataFrame):
            if "vehicle_count" in X.columns:
                values = X["vehicle_count"]
            else:
                values = X.iloc[:, self._column_index]
        else:
            values = np.asarray(X)[:, self._column_index]
        return np.asarray(values, dtype=float)
