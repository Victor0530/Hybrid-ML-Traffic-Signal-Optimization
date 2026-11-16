"""Contract tests for shared SL data access, preprocessing and model plugins."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.preprocessing.features import DIRECTIONS
from src.supervised.data import load_splits
from src.supervised.pipeline import fit_preprocessor, transform
from src.supervised.predictor import fit_named_predictor
from src.supervised.registry import (
    available_models,
    build_model,
    register_model,
)


def synthetic_dataset() -> pd.DataFrame:
    rows = []
    for split in ("train", "validation", "test"):
        for index, direction in enumerate(DIRECTIONS):
            row = {
                "split": split,
                "direction": direction,
                "target_vehicle_count": float(index + 1),
                "vehicle_count": float(index),
                "no_vehicles": 0,
                "sensor_missing": 0,
                **{f"direction_{name}": int(name == direction) for name in DIRECTIONS},
            }
            rows.append(row)
    return pd.DataFrame(rows)


def test_load_splits_preserves_feature_order(tmp_path: Path) -> None:
    path = tmp_path / "dataset.csv"
    synthetic_dataset().to_csv(path, index=False)

    splits = load_splits(feature_set="F0", path=path)

    assert set(splits) == {"train", "validation", "test"}
    assert splits["train"].n_samples == 4
    assert splits["train"].feature_names[0] == "vehicle_count"
    assert list(splits["validation"].X.columns) == list(
        splits["train"].X.columns
    )
    assert not splits["test"].y.isna().any()


def test_preprocessor_is_fitted_on_training_rows_only() -> None:
    X_train = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    X_validation = pd.DataFrame({"a": [np.nan, 100.0], "b": [1.0, 2.0]})

    preprocessor = fit_preprocessor(X_train)
    transformed = transform(preprocessor, X_validation)

    assert preprocessor.named_steps["imputer"].statistics_[0] == 2.0
    assert np.isfinite(transformed).all()


def test_registry_builds_the_five_candidates() -> None:
    assert set(available_models()) == {
        "decision_tree",
        "hist_gradient_boosting",
        "random_forest",
        "ridge",
        "xgboost",
    }
    assert isinstance(build_model("ridge"), Ridge)
    assert isinstance(build_model("decision_tree"), DecisionTreeRegressor)
    assert isinstance(build_model("random_forest"), RandomForestRegressor)
    assert isinstance(
        build_model("hist_gradient_boosting"), HistGradientBoostingRegressor
    )
    assert isinstance(build_model("xgboost"), XGBRegressor)
    assert build_model("ridge", {"alpha": 5.0}).alpha == 5.0
    with pytest.raises(KeyError):
        build_model("not_a_model")


def test_registry_accepts_new_plugins() -> None:
    register_model("test_model", DecisionTreeRegressor, {"max_depth": 2})
    try:
        model = build_model("test_model")
        assert model.max_depth == 2
    finally:
        from src.supervised import registry

        registry._CANDIDATES.pop("test_model", None)


def test_predictor_returns_one_value_per_direction(tmp_path: Path) -> None:
    frame = synthetic_dataset()
    train = frame.loc[frame["split"] == "train"]
    features = [
        "vehicle_count",
        "no_vehicles",
        "sensor_missing",
        *[f"direction_{name}" for name in DIRECTIONS],
    ]
    predictor = fit_named_predictor(
        "ridge",
        train.loc[:, features],
        train["target_vehicle_count"],
        metadata={"feature_set": "F0"},
    )

    window = frame.loc[frame["split"] == "test", ["direction", *features]].copy()
    predictions = predictor.predict(window)

    assert list(predictions) == list(DIRECTIONS)
    assert all(np.isfinite(value) for value in predictions.values())

    path = tmp_path / "predictor.joblib"
    predictor.save(path)
    loaded = type(predictor).load(path)
    assert loaded.predict(window) == pytest.approx(predictions)
