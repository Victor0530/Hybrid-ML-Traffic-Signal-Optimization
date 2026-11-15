"""Serializable prediction interface used by the hybrid controller.

A :class:`Predictor` bundles the fitted train-only preprocessing pipeline, the
model and the exact feature order. The hybrid stage only needs to call
``predict`` with four current direction rows and receives one future count per
direction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from src.preprocessing.features import DIRECTIONS
from src.supervised.pipeline import build_model_pipeline


class Predictor:
    """Fitted SL model plus preprocessing and feature-order metadata."""

    def __init__(
        self,
        pipeline: Any,
        feature_names: tuple[str, ...],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.feature_names = tuple(feature_names)
        self.metadata = dict(metadata or {})

    def predict_array(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict one value per input row."""
        if isinstance(X, pd.DataFrame):
            missing = [
                column for column in self.feature_names if column not in X.columns
            ]
            if missing:
                raise ValueError(
                    "Prediction input is missing features: " + ", ".join(missing)
                )
            values = X.loc[:, self.feature_names]
        else:
            array = np.asarray(X, dtype=float)
            if array.ndim != 2 or array.shape[1] != len(self.feature_names):
                raise ValueError(
                    "Prediction array must have shape "
                    f"(n_rows, {len(self.feature_names)})"
                )
            values = array
        return np.asarray(self.pipeline.predict(values), dtype=float)

    def predict(self, window: pd.DataFrame) -> dict[str, float]:
        """Predict future counts for a window containing all four directions.

        ``window`` must contain one row per direction and a ``direction``
        column. The returned dictionary always follows the north, south, east,
        west order.
        """
        if "direction" not in window.columns:
            raise ValueError("Prediction window must contain a 'direction' column")
        directions = list(window["direction"])
        if sorted(directions) != sorted(DIRECTIONS):
            raise ValueError(
                "Prediction window must contain exactly one row per direction: "
                f"{DIRECTIONS}"
            )
        predictions = self.predict_array(window)
        by_direction = {
            str(direction): float(value)
            for direction, value in zip(directions, predictions, strict=True)
        }
        return {direction: by_direction[direction] for direction in DIRECTIONS}

    def save(self, path: str | Path) -> Path:
        """Persist the predictor as one joblib artifact."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "feature_names": self.feature_names,
                "metadata": self.metadata,
            },
            output,
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "Predictor":
        payload = joblib.load(Path(path))
        return cls(
            pipeline=payload["pipeline"],
            feature_names=tuple(payload["feature_names"]),
            metadata=payload.get("metadata", {}),
        )


def fit_predictor(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> Predictor:
    """Fit the shared preprocessing + model pipeline on training rows only."""
    pipeline = build_model_pipeline(model, scale=scale)
    pipeline.fit(X_train, y_train)
    return Predictor(
        pipeline=pipeline,
        feature_names=tuple(X_train.columns),
        metadata=metadata,
    )


def fit_named_predictor(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Predictor:
    """Fit a registered model while respecting its scaling requirement."""
    from src.supervised.registry import build_model, model_spec

    spec = model_spec(name)
    model = build_model(name, dict(params or {}))
    return fit_predictor(
        model,
        X_train,
        y_train,
        scale=spec.scale,
        metadata={
            "model": spec.name,
            "feature_names": list(X_train.columns),
            **(metadata or {}),
        },
    )
