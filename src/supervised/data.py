"""Stable data-access contract for supervised-learning teammates.

Every SL model must load the frozen dataset through this module. It returns
the same feature order and the same train/validation/test partitions, so model
comparisons cannot accidentally use different rows or columns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocessing.features import (
    DEFAULT_CONFIG,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    feature_columns,
    load_sl_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitData:
    """One chronological split, with aligned features, target and metadata."""

    name: str
    X: pd.DataFrame
    y: pd.Series
    frame: pd.DataFrame
    feature_names: tuple[str, ...]

    @property
    def n_samples(self) -> int:
        return len(self.frame)


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _config_dict(config: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    return load_sl_config(config or DEFAULT_CONFIG)


def dataset_path(
    config: dict[str, Any] | str | Path | None = None,
) -> Path:
    """Return the configured processed-dataset path."""
    cfg = _config_dict(config)
    directory = _resolve_path(cfg["data"]["processed_dir"])
    return directory / cfg["output"]["dataset"]


def load_dataset(
    config: dict[str, Any] | str | Path | None = None,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the complete processed dataset without modifying it."""
    dataset_file = _resolve_path(path) if path is not None else dataset_path(config)
    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {dataset_file}. "
            "Run `python -m src.preprocessing.build_dataset` first."
        )
    return pd.read_csv(dataset_file)


def load_feature_map(
    config: dict[str, Any] | str | Path | None = None,
    path: str | Path | None = None,
) -> dict[str, list[str]]:
    """Load the generated feature-set map, mainly for provenance checks."""
    cfg = _config_dict(config)
    feature_file = (
        _resolve_path(path)
        if path is not None
        else _resolve_path(cfg["data"]["processed_dir"]) / cfg["output"]["feature_map"]
    )
    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature map not found: {feature_file}. "
            "Run `python -m src.preprocessing.build_dataset` first."
        )
    with feature_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_splits(
    config: dict[str, Any] | str | Path | None = None,
    feature_set: str | None = None,
    path: str | Path | None = None,
) -> dict[str, SplitData]:
    """Load train/validation/test frames through the shared contract.

    The target must be complete. No imputation, scaling or shuffling happens
    here; those belong in the train-only pipeline.
    """
    cfg = _config_dict(config)
    feature_set = feature_set or cfg["features"]["default_set"]
    features = feature_columns(feature_set)
    frame = load_dataset(cfg, path)

    required = {SPLIT_COLUMN, TARGET_COLUMN, *features}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )

    splits: dict[str, SplitData] = {}
    for split_name in SPLIT_NAMES:
        part = frame.loc[frame[SPLIT_COLUMN] == split_name].copy()
        if part.empty:
            raise ValueError(f"Split {split_name!r} is empty")
        if part[TARGET_COLUMN].isna().any():
            raise ValueError(f"Split {split_name!r} contains an undefined target")
        splits[split_name] = SplitData(
            name=split_name,
            X=part.loc[:, features].copy(),
            y=part.loc[:, TARGET_COLUMN].astype(float).copy(),
            frame=part,
            feature_names=features,
        )
    return splits
