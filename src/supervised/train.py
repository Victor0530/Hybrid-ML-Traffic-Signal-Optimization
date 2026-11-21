"""Shared supervised-learning training entry point.

Trains one or more registered candidates on the frozen train split, reports
metrics on the validation split only (the test split is never touched here)
and optionally saves the fitted predictor for later evaluation.

Usage:
    python -m src.supervised.train --config configs/sl.yaml
    python -m src.supervised.train --models ridge persistence --feature-set F0
    python -m src.supervised.train --models random_forest --tune

Tuning uses expanding time-block cross-validation inside the train split:
each fold trains on simulation times <= E and validates on the contiguous
300 s window that starts 305 s later, so every fold keeps all 20 runs and
no training target can overlap a validation input (the same 300 s embargo
rule the frozen splits use). ``--seed`` is passed to every seeded model as
``random_state`` so runs are repeatable and seed-sensitive.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.evaluation.metrics import regression_metrics
from src.preprocessing.features import load_sl_config
from src.supervised import (
    Predictor,
    available_models,
    fit_named_predictor,
    load_splits,
)
from src.supervised.pipeline import build_model_pipeline
from src.supervised.registry import build_model, model_spec
from src.utils import set_global_seed

DEFAULT_CONFIG = Path("configs/sl.yaml")


def time_block_folds(
    frame: pd.DataFrame,
    n_splits: int = 3,
    gap_seconds: int = 300,
    window_seconds: int = 300,
    interval_seconds: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV over ``simulation_time`` (all runs in every fold).

    The last validation window ends at the last training instant; earlier
    windows step back by ``window_seconds``. Each fold trains on rows at
    most ``train_end = val_end - gap - window`` seconds and validates on
    the ``window_seconds`` window ending at ``val_end``, which keeps a gap
    of at least ``gap_seconds`` between the newest training target and the
    oldest validation input.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be positive")
    times = frame["simulation_time"].to_numpy(dtype=float)
    if times.size == 0:
        raise ValueError("Cannot build folds from an empty frame")
    min_time = float(times.min())
    max_time = float(times.max())

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for split in range(n_splits):
        val_end = max_time - (n_splits - 1 - split) * window_seconds
        val_start = val_end - window_seconds + interval_seconds
        train_end = val_end - gap_seconds - window_seconds
        if train_end < min_time:
            raise ValueError(
                f"Not enough training time for fold {split + 1}: train must "
                f"end at or after {min_time}s but the earliest allowed end "
                f"is {train_end}s (reduce n_splits or the embargo window)"
            )
        train_idx = np.flatnonzero(times <= train_end)
        val_idx = np.flatnonzero((times >= val_start) & (times <= val_end))
        if train_idx.size == 0 or val_idx.size == 0:
            raise ValueError(f"Fold {split + 1} produced an empty split")
        folds.append((train_idx, val_idx))
    return folds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--feature-set", default=None, help="default: config features.default_set")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="registered model names; default: every candidate plus the persistence baseline",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune", action="store_true", help="grid-search on the train split with time-aware CV")
    parser.add_argument("--output-dir", type=Path, default=Path("results/sl/metrics"))
    parser.add_argument("--save-dir", type=Path, default=Path("models/sl"))
    parser.add_argument("--no-save", action="store_true", help="skip writing predictor artifacts")
    return parser.parse_args(argv)


def _grid_for(name: str, config: dict[str, Any]) -> dict[str, list[Any]]:
    block = (config.get("models") or {}).get(name) or {}
    grid = block.get("grid") or {}
    # GridSearchCV operates on the full pipeline, so estimator parameters
    # must use the pipeline step prefix ("model__").
    return {f"model__{key}": value for key, value in grid.items()}


def _params_for(name: str, config: dict[str, Any]) -> dict[str, Any]:
    block = (config.get("models") or {}).get(name) or {}
    return dict(block.get("params") or {})


def _effective_params(name: str, config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Config params plus the CLI seed for models that accept random_state."""
    params = _params_for(name, config)
    if model_spec(name).seeded:
        params = {**params, "random_state": int(seed)}
    return params


def _fit_one(
    name: str,
    splits: dict[str, Any],
    config: dict[str, Any],
    feature_set: str,
    seed: int,
    tune: bool,
) -> tuple[dict[str, Any], Predictor]:
    """Fit one model on train, score on validation, return its metrics record."""
    train, validation = splits["train"], splits["validation"]
    params = _effective_params(name, config, seed)
    spec = model_spec(name)
    started = time.perf_counter()
    best_params: dict[str, Any] | None = None

    grid = _grid_for(name, config) if tune else {}
    if tune and not grid:
        raise ValueError(
            f"Model {name!r} has no tuning grid in the configuration; "
            "add a models.<name>.grid block or drop --tune"
        )

    if grid:
        pipeline = build_model_pipeline(build_model(name, params), scale=spec.scale)
        search = GridSearchCV(
            pipeline,
            grid,
            cv=time_block_folds(train.frame),
            scoring="neg_mean_absolute_error",
            n_jobs=1,  # estimators such as random_forest already use all cores
            refit=True,
        )
        search.fit(train.X, train.y)
        predictor = Predictor(
            pipeline=search.best_estimator_,
            feature_names=tuple(train.X.columns),
            metadata={
                "model": name,
                "feature_set": feature_set,
                "tuned": True,
                "best_params": {
                    key.removeprefix("model__"): value
                    for key, value in search.best_params_.items()
                },
            },
        )
        best_params = predictor.metadata["best_params"]
    else:
        predictor = fit_named_predictor(
            name,
            train.X,
            train.y,
            params=params or None,
            metadata={"feature_set": feature_set, "tuned": False},
        )

    fit_seconds = time.perf_counter() - started
    train_pred = predictor.pipeline.predict(train.X)
    valid_pred = predictor.pipeline.predict(validation.X)
    return {
        "model": name,
        "feature_set": feature_set,
        "seed": seed,
        "tuned": bool(grid),
        "best_params": best_params,
        "params": params,
        "n_train": int(len(train.X)),
        "n_validation": int(len(validation.X)),
        "train": regression_metrics(train.y, train_pred),
        "validation": regression_metrics(validation.y, valid_pred),
        "fit_seconds": round(fit_seconds, 3),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, predictor


def train_models(
    config: dict[str, Any],
    feature_set: str,
    model_names: list[str],
    seed: int = 42,
    tune: bool = False,
    output_dir: Path = Path("results/sl/metrics"),
    save_dir: Path | None = Path("models/sl"),
    save: bool = True,
) -> list[dict[str, Any]]:
    """Train every requested model and persist metrics (and predictors)."""
    set_global_seed(seed)
    splits = load_splits(config, feature_set=feature_set)

    records: list[dict[str, Any]] = []
    for name in model_names:
        record, predictor = _fit_one(name, splits, config, feature_set, seed, tune)
        records.append(record)

        metrics_dir = Path(output_dir) / feature_set
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with (metrics_dir / f"{name}.json").open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)

        if save and save_dir is not None:
            predictor.save(Path(save_dir) / feature_set / name / "predictor.joblib")

        print(
            f"{name:>24}  val MAE {record['validation']['mae']:8.4f}  "
            f"RMSE {record['validation']['rmse']:8.4f}  "
            f"R2 {record['validation']['r2']:7.4f}  "
            f"({record['fit_seconds']}s)"
        )

    # Ranking table over everything trained in this invocation.
    summary = pd.DataFrame(
        [
            {
                "model": rec["model"],
                "feature_set": rec["feature_set"],
                "tuned": rec["tuned"],
                "val_mae": rec["validation"]["mae"],
                "val_rmse": rec["validation"]["rmse"],
                "val_r2": rec["validation"]["r2"],
                "val_mape": rec["validation"]["mape"],
                "fit_seconds": rec["fit_seconds"],
            }
            for rec in records
        ]
    ).sort_values("val_mae")
    metrics_dir = Path(output_dir) / feature_set
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = metrics_dir / "summary.csv"
    if summary_path.exists():
        previous = pd.read_csv(summary_path)
        merged = pd.concat([previous, summary], ignore_index=True)
        merged = merged.drop_duplicates(subset=["model", "feature_set", "tuned"], keep="last")
        merged = merged.sort_values("val_mae")
        merged.to_csv(summary_path, index=False)
    else:
        summary.to_csv(summary_path, index=False)
    return records


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_sl_config(args.config)
    feature_set = args.feature_set or config["features"]["default_set"]
    model_names = args.models or list(available_models(include_baselines=True))
    train_models(
        config=config,
        feature_set=feature_set,
        model_names=model_names,
        seed=args.seed,
        tune=args.tune,
        output_dir=args.output_dir,
        save_dir=None if args.no_save else args.save_dir,
        save=not args.no_save,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
