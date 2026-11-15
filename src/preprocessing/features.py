"""Build the supervised-learning dataset from raw traffic observations.

The raw collector writes one long-format CSV per run with one row per
``(simulation_time, direction)``. This module:

1. loads the successful runs recorded in the run manifest;
2. validates timestamp continuity and direction completeness;
3. builds the 5-minute-ahead target per ``(run_id, direction)``;
4. derives causal lag, rolling, congestion, signal and imbalance features;
5. assigns leakage-safe chronological train/validation/test splits with an
   embargo of one prediction horizon between blocks; and
6. writes the dataset, split files, feature map and a data-quality audit.

Nothing in this module fits imputers or scalers: that happens on the training
split only, inside the model pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sl.yaml"

DIRECTIONS = ("north", "south", "east", "west")
FEATURE_SETS = ("F0", "F1", "F2", "F3", "F4")
TARGET_COLUMN = "target_vehicle_count"
SPLIT_COLUMN = "split"
DEFAULT_PHASES = tuple(range(8))

REQUIRED_OBSERVATION_COLUMNS = {
    "run_id",
    "scenario",
    "seed",
    "simulation_time",
    "direction",
    "vehicle_count",
    "mean_speed",
    "queue_length",
    "waiting_time",
    "occupancy",
    "traffic_light_phase",
    "phase_duration",
    "time_in_phase",
    "no_vehicles",
    "sensor_missing",
    "departed_count",
    "arrived_count",
}

NUMERIC_OBSERVATION_COLUMNS = (
    "simulation_time",
    "vehicle_count",
    "mean_speed",
    "queue_length",
    "waiting_time",
    "occupancy",
    "traffic_light_phase",
    "phase_duration",
    "time_in_phase",
    "no_vehicles",
    "sensor_missing",
    "departed_count",
    "arrived_count",
)

# Feature-set definitions. Every set includes the F0 columns so the ablation is
# strictly incremental (F0 subset F1 subset ... subset F4).
COUNT_LAGS = (1, 3, 6, 12, 36, 60)  # 5 s ... 300 s
COUNT_ROLLING = (12, 36, 60)  # 1 min, 3 min, 5 min
CONGESTION_ROLLING = (12,)  # 1 min

F0_COLUMNS = (
    "vehicle_count",
    "no_vehicles",
    "sensor_missing",
    *(f"direction_{direction}" for direction in DIRECTIONS),
)
F1_COLUMNS = (
    *(f"count_lag_{lag}" for lag in COUNT_LAGS),
    *(f"count_roll_mean_{window}" for window in COUNT_ROLLING),
    *(f"count_roll_std_{window}" for window in COUNT_ROLLING),
)
F2_COLUMNS = (
    "queue_length",
    "mean_speed",
    "waiting_time",
    "occupancy",
    *(f"queue_roll_mean_{window}" for window in CONGESTION_ROLLING),
    *(f"speed_roll_mean_{window}" for window in CONGESTION_ROLLING),
    *(f"waiting_roll_mean_{window}" for window in CONGESTION_ROLLING),
)
F3_COLUMNS = (
    *(f"phase_{phase}" for phase in DEFAULT_PHASES),
    "time_in_phase",
    "phase_duration",
)
F4_COLUMNS = (
    "count_total",
    "count_range",
    "count_share",
    "queue_total",
    "queue_range",
)

FEATURE_SET_COLUMNS = {
    "F0": tuple(F0_COLUMNS),
    "F1": tuple(F0_COLUMNS) + tuple(F1_COLUMNS),
    "F2": tuple(F0_COLUMNS) + tuple(F1_COLUMNS) + tuple(F2_COLUMNS),
    "F3": tuple(F0_COLUMNS) + tuple(F1_COLUMNS) + tuple(F2_COLUMNS) + tuple(F3_COLUMNS),
    "F4": tuple(F0_COLUMNS)
    + tuple(F1_COLUMNS)
    + tuple(F2_COLUMNS)
    + tuple(F3_COLUMNS)
    + tuple(F4_COLUMNS),
}


def load_sl_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_traffic_config(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def feature_columns(feature_set: str) -> tuple[str, ...]:
    """Return the ordered feature columns for a named feature set."""
    try:
        return FEATURE_SET_COLUMNS[feature_set]
    except KeyError as exc:
        raise ValueError(
            f"Unknown feature set {feature_set!r}; choose from {FEATURE_SETS}."
        ) from exc


def load_observations(
    raw_dir: Path | str,
    manifest_path: Path | str,
    duration: int,
    interval: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load observation CSVs for runs marked successful in the manifest."""
    raw_dir = Path(raw_dir)
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing run manifest: {manifest_path}. Run the M2 collector first."
        )

    manifest = pd.read_csv(manifest_path)
    required_manifest = {
        "run_id",
        "scenario",
        "seed",
        "status",
        "duration",
        "sampling_interval",
        "simulation_end",
    }
    missing = required_manifest - set(manifest.columns)
    if missing:
        raise ValueError(f"Run manifest is missing columns: {sorted(missing)}")

    runs = manifest[
        (manifest["status"] == "ok")
        & (manifest["duration"] == duration)
        & (manifest["sampling_interval"] == interval)
    ].copy()
    if runs.empty:
        raise ValueError(
            "No successful runs match the configured duration and sampling interval."
        )

    frames: list[pd.DataFrame] = []
    for row in runs.itertuples(index=False):
        path = raw_dir / f"{row.run_id}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Manifest lists {row.run_id}, but {path} is missing.")
        frame = pd.read_csv(path)
        cols = REQUIRED_OBSERVATION_COLUMNS - set(frame.columns)
        if cols:
            raise ValueError(
                f"{path.name} is missing columns: {sorted(cols)}. "
                "Rerun the collector with --overwrite."
            )
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    for column in NUMERIC_OBSERVATION_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data, runs


def prepare_observations(
    data: pd.DataFrame,
    duration: int,
    interval: int,
) -> pd.DataFrame:
    """Validate and normalise raw observations, sorted by run/direction/time."""
    missing = REQUIRED_OBSERVATION_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Observations are missing columns: {sorted(missing)}")

    prepared = data.copy()
    for column in NUMERIC_OBSERVATION_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.sort_values(
        ["run_id", "direction", "simulation_time"], kind="stable"
    ).reset_index(drop=True)

    duplicates = int(
        prepared.duplicated(["run_id", "simulation_time", "direction"]).sum()
    )
    if duplicates:
        raise ValueError(
            f"Found {duplicates} duplicate (run_id, simulation_time, direction) rows."
        )

    unknown = sorted(set(prepared["direction"]) - set(DIRECTIONS))
    if unknown:
        raise ValueError(f"Unknown direction values: {unknown}")

    expected_samples = duration // interval
    if duration % interval:
        raise ValueError(
            f"Duration {duration}s is not a multiple of interval {interval}s."
        )

    continuity_failures: list[str] = []
    incomplete_groups: list[str] = []
    for (run_id, direction), group in prepared.groupby(
        ["run_id", "direction"], sort=False
    ):
        times = group["simulation_time"].to_numpy()
        if len(times) != expected_samples:
            incomplete_groups.append(
                f"{run_id}/{direction}: {len(times)} samples, expected {expected_samples}"
            )
        if len(times) > 1 and not np.all(np.diff(times) == interval):
            continuity_failures.append(f"{run_id}/{direction}")

    if incomplete_groups:
        raise ValueError(
            "Incomplete run/direction groups: " + "; ".join(incomplete_groups[:5])
        )
    if continuity_failures:
        raise ValueError(
            "Non-contiguous timestamps in: " + "; ".join(continuity_failures[:5])
        )

    per_timestamp = prepared.groupby(["run_id", "simulation_time"]).size()
    incomplete_timestamps = per_timestamp[per_timestamp != len(DIRECTIONS)]
    if not incomplete_timestamps.empty:
        raise ValueError(
            f"{len(incomplete_timestamps)} timestamps do not have all "
            f"{len(DIRECTIONS)} directions."
        )

    prepared["sample_index"] = prepared.groupby(
        ["run_id", "direction"], sort=False
    ).cumcount()
    return prepared


def add_target(data: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
    """Add the future vehicle count per run/direction without imputing it."""
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive.")
    ordered = data.sort_values(
        ["run_id", "direction", "simulation_time"], kind="stable"
    ).reset_index(drop=True)
    ordered[TARGET_COLUMN] = ordered.groupby(
        ["run_id", "direction"], sort=False
    )["vehicle_count"].shift(-horizon_steps)
    return ordered


def _rolling_mean(
    grouped: pd.core.groupby.generic.SeriesGroupBy,
    window: int,
    min_periods: int,
) -> pd.Series:
    return grouped.transform(
        lambda values: values.rolling(window, min_periods=min_periods).mean()
    )


def _rolling_std(
    grouped: pd.core.groupby.generic.SeriesGroupBy,
    window: int,
    min_periods: int,
) -> pd.Series:
    return grouped.transform(
        lambda values: values.rolling(window, min_periods=min_periods).std()
    )


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Derive every feature column; selection happens via ``feature_columns``.

    All historical features are causal: lag ``k`` uses ``t - k * interval`` and
    rolling windows end at the current sample. The current interval is allowed
    because it is observed before the prediction is made.
    """
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Call add_target() before build_features(); missing {TARGET_COLUMN}.")

    featured = data.copy()

    for direction in DIRECTIONS:
        featured[f"direction_{direction}"] = (
            featured["direction"] == direction
        ).astype("int8")
    for phase in DEFAULT_PHASES:
        featured[f"phase_{phase}"] = (
            featured["traffic_light_phase"] == phase
        ).astype("int8")

    grouped = featured.groupby(["run_id", "direction"], sort=False)

    for lag in COUNT_LAGS:
        featured[f"count_lag_{lag}"] = grouped["vehicle_count"].shift(lag)
    for window in COUNT_ROLLING:
        featured[f"count_roll_mean_{window}"] = _rolling_mean(
            grouped["vehicle_count"], window, window
        )
        featured[f"count_roll_std_{window}"] = _rolling_std(
            grouped["vehicle_count"], window, window
        )

    for window in CONGESTION_ROLLING:
        featured[f"queue_roll_mean_{window}"] = _rolling_mean(
            grouped["queue_length"], window, window
        )
        featured[f"speed_roll_mean_{window}"] = _rolling_mean(
            grouped["mean_speed"], window, 1
        )
        featured[f"waiting_roll_mean_{window}"] = _rolling_mean(
            grouped["waiting_time"], window, window
        )

    # Cross-direction context. Each timestamp has exactly four rows, one per
    # approach, so these describe the other approaches without leaking time.
    by_timestamp = featured.groupby(["run_id", "simulation_time"], sort=False)
    featured["count_total"] = by_timestamp["vehicle_count"].transform("sum")
    featured["count_range"] = by_timestamp["vehicle_count"].transform(
        lambda values: values.max() - values.min()
    )
    featured["count_share"] = np.where(
        featured["count_total"] > 0,
        featured["vehicle_count"] / featured["count_total"],
        0.0,
    )
    featured["queue_total"] = by_timestamp["queue_length"].transform("sum")
    featured["queue_range"] = by_timestamp["queue_length"].transform(
        lambda values: values.max() - values.min()
    )

    return featured


def assign_splits(
    data: pd.DataFrame,
    split_config: dict[str, Any],
    duration: int,
    horizon_seconds: int,
) -> pd.DataFrame:
    """Assign chronological splits with an embargo between adjacent blocks."""
    train_end = int(split_config["train_end"])
    validation_start = int(split_config["validation_start"])
    validation_end = int(split_config["validation_end"])
    test_start = int(split_config["test_start"])
    test_end = int(split_config["test_end"])
    embargo = int(split_config["embargo_seconds"])

    expected_test_end = duration - horizon_seconds
    if test_end != expected_test_end:
        raise ValueError(
            f"test_end={test_end}s must equal duration - horizon = {expected_test_end}s."
        )
    if embargo < horizon_seconds:
        raise ValueError(
            f"embargo_seconds={embargo}s must be at least the prediction horizon "
            f"({horizon_seconds}s) to avoid shared targets across splits."
        )
    if validation_start - train_end < embargo:
        raise ValueError("The train/validation embargo is shorter than configured.")
    if test_start - validation_end < embargo:
        raise ValueError("The validation/test embargo is shorter than configured.")
    if not train_end < validation_start <= validation_end < test_start <= test_end:
        raise ValueError("Split boundaries are not strictly ordered.")

    assigned = data.copy()
    assigned[SPLIT_COLUMN] = "excluded"
    assigned.loc[assigned["simulation_time"] <= train_end, SPLIT_COLUMN] = "train"
    assigned.loc[
        (assigned["simulation_time"] >= validation_start)
        & (assigned["simulation_time"] <= validation_end),
        SPLIT_COLUMN,
    ] = "validation"
    assigned.loc[
        (assigned["simulation_time"] >= test_start)
        & (assigned["simulation_time"] <= test_end),
        SPLIT_COLUMN,
    ] = "test"
    return assigned


def build_dataset(
    config: dict[str, Any] | None = None,
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build, write and return the SL dataset, splits, feature map and audit."""
    if config is None:
        config = load_sl_config()

    data_config = config["data"]
    traffic_config_path = Path(config["traffic_config"])
    if not traffic_config_path.is_absolute():
        traffic_config_path = PROJECT_ROOT / traffic_config_path
    traffic = load_traffic_config(traffic_config_path)

    simulation = traffic["simulation"]
    duration = int(simulation["duration"])
    interval = int(simulation["sampling_interval"])
    horizon_seconds = int(simulation["prediction_horizon"])
    horizon_steps = horizon_seconds // interval
    warmup_seconds = int(config["features"]["warmup_seconds"])
    warmup_samples = warmup_seconds // interval

    raw_path = Path(raw_dir) if raw_dir is not None else PROJECT_ROOT / data_config["raw_dir"]
    output_path = (
        Path(processed_dir)
        if processed_dir is not None
        else PROJECT_ROOT / data_config["processed_dir"]
    )

    raw, runs = load_observations(
        raw_path,
        raw_path / Path(data_config["manifest"]).name,
        duration=duration,
        interval=interval,
    )
    prepared = prepare_observations(raw, duration=duration, interval=interval)
    with_target = add_target(prepared, horizon_steps=horizon_steps)

    # Derive features on the full timeline before dropping rows: lag and rolling
    # history must not be amputated by the warm-up or target filters.
    featured = build_features(with_target)

    target_missing = int(featured[TARGET_COLUMN].isna().sum())
    dataset = featured.dropna(subset=[TARGET_COLUMN]).copy()

    warmup_rows = int((dataset["sample_index"] < warmup_samples).sum())
    dataset = dataset[dataset["sample_index"] >= warmup_samples].copy()

    dataset = assign_splits(
        dataset,
        split_config=config["split"],
        duration=duration,
        horizon_seconds=horizon_seconds,
    )

    excluded_rows = int((dataset[SPLIT_COLUMN] == "excluded").sum())
    dataset = dataset[dataset[SPLIT_COLUMN] != "excluded"].copy()
    dataset = dataset.sort_values(
        ["run_id", "simulation_time", "direction"], kind="stable"
    ).reset_index(drop=True)

    audit = build_audit(
        raw=raw,
        runs=runs,
        dataset=dataset,
        duration=duration,
        interval=interval,
        horizon_seconds=horizon_seconds,
        target_missing=target_missing,
        warmup_rows=warmup_rows,
        excluded_rows=excluded_rows,
    )

    splits_dir = output_path / Path(data_config["splits_dir"]).name
    output_path.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_path / config["output"]["dataset"]
    dataset.to_csv(dataset_path, index=False)

    split_counts: dict[str, int] = {}
    for split_name in ("train", "validation", "test"):
        split = dataset[dataset[SPLIT_COLUMN] == split_name].copy()
        split_counts[split_name] = int(len(split))
        split.to_csv(splits_dir / f"{split_name}.csv", index=False)
    audit["split_rows"] = split_counts

    feature_map = {
        feature_set: list(columns) for feature_set, columns in FEATURE_SET_COLUMNS.items()
    }
    feature_map_path = output_path / config["output"]["feature_map"]
    feature_map_path.write_text(
        json.dumps(feature_map, indent=2) + "\n", encoding="utf-8"
    )

    audit_path = output_path / config["output"]["audit"]
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    return dataset, audit


def build_audit(
    raw: pd.DataFrame,
    runs: pd.DataFrame,
    dataset: pd.DataFrame,
    duration: int,
    interval: int,
    horizon_seconds: int,
    target_missing: int,
    warmup_rows: int,
    excluded_rows: int,
) -> dict[str, Any]:
    """Summarise data quality, target availability and split coverage."""
    zero_vehicle = (
        dataset.groupby("direction")["no_vehicles"].mean().sort_index() * 100.0
    )
    missing_by_column = {
        column: int(dataset[column].isna().sum())
        for column in dataset.columns
        if int(dataset[column].isna().sum()) > 0
    }
    return {
        "runs": int(runs["run_id"].nunique()),
        "scenarios": sorted(dataset["scenario"].unique().tolist()),
        "raw_rows": int(len(raw)),
        "dataset_rows": int(len(dataset)),
        "target_undefined_rows": target_missing,
        "warmup_rows_dropped": warmup_rows,
        "embargo_excluded_rows": excluded_rows,
        "split_rows": {},
        "rows_by_run": {
            str(run_id): int(count)
            for run_id, count in dataset.groupby("run_id").size().items()
        },
        "missing_values_by_column": missing_by_column,
        "zero_vehicle_pct_by_direction": {
            direction: round(float(value), 3)
            for direction, value in zero_vehicle.items()
        },
        "duplicate_keys": int(
            dataset.duplicated(["run_id", "simulation_time", "direction"]).sum()
        ),
        "config": {
            "duration": duration,
            "sampling_interval": interval,
            "prediction_horizon": horizon_seconds,
            "horizon_steps": horizon_seconds // interval,
        },
        "feature_counts": {
            feature_set: len(columns)
            for feature_set, columns in FEATURE_SET_COLUMNS.items()
        },
    }
