"""Tests for the supervised-learning dataset, target and split construction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.features import (
    DIRECTIONS,
    FEATURE_SETS,
    FEATURE_SET_COLUMNS,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    add_target,
    assign_splits,
    build_dataset,
    build_features,
    feature_columns,
    prepare_observations,
)

DURATION = 3600
INTERVAL = 5
HORIZON = 300
HORIZON_STEPS = HORIZON // INTERVAL
WARMUP_SAMPLES = HORIZON_STEPS

SPLIT_CONFIG = {
    "train_end": 1800,
    "validation_start": 2105,
    "validation_end": 2700,
    "test_start": 3005,
    "test_end": 3300,
    "embargo_seconds": 300,
}


def make_observations(
    runs: tuple[str, ...] = ("run_a",),
    duration: int = DURATION,
    interval: int = INTERVAL,
    count_fn=None,
) -> pd.DataFrame:
    """Create long-format observations that satisfy the collector schema."""
    records: list[dict[str, object]] = []
    samples = duration // interval
    for run_id in runs:
        for index in range(samples):
            simulation_time = (index + 1) * interval
            for direction_index, direction in enumerate(DIRECTIONS):
                count = (
                    index
                    if count_fn is None
                    else int(count_fn(index, direction))
                )
                no_vehicles = int(count == 0)
                speed = None if no_vehicles else 10.0 + direction_index * 0.5
                records.append(
                    {
                        "run_id": run_id,
                        "scenario": "medium",
                        "seed": 42,
                        "simulation_time": simulation_time,
                        "direction": direction,
                        "vehicle_count": count,
                        "mean_speed": speed,
                        "queue_length": count % 4,
                        "waiting_time": float(count),
                        "occupancy": 0.01 * count,
                        "traffic_light_phase": index % 8,
                        "phase_duration": 30.0,
                        "time_in_phase": float(index % 30),
                        "no_vehicles": no_vehicles,
                        "sensor_missing": 0,
                        "departed_count": count,
                        "arrived_count": count,
                    }
                )
    return pd.DataFrame.from_records(records)


def build_labelled_dataset(data: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_observations(data, duration=DURATION, interval=INTERVAL)
    featured = build_features(add_target(prepared, horizon_steps=HORIZON_STEPS))
    dataset = featured.dropna(subset=[TARGET_COLUMN]).copy()
    dataset = dataset[dataset["sample_index"] >= WARMUP_SAMPLES].copy()
    return assign_splits(
        dataset,
        split_config=SPLIT_CONFIG,
        duration=DURATION,
        horizon_seconds=HORIZON,
    )


def test_target_is_exactly_one_horizon_ahead() -> None:
    prepared = prepare_observations(
        make_observations(count_fn=lambda index, _direction: index),
        duration=DURATION,
        interval=INTERVAL,
    )
    labelled = add_target(prepared, horizon_steps=HORIZON_STEPS)

    row = labelled[
        (labelled["run_id"] == "run_a")
        & (labelled["direction"] == "north")
        & (labelled["simulation_time"] == 300)
    ].iloc[0]
    future = prepared[
        (prepared["run_id"] == "run_a")
        & (prepared["direction"] == "north")
        & (prepared["simulation_time"] == 600)
    ].iloc[0]

    assert row[TARGET_COLUMN] == future["vehicle_count"]


def test_last_horizon_samples_have_no_target() -> None:
    prepared = prepare_observations(
        make_observations(), duration=DURATION, interval=INTERVAL
    )
    labelled = add_target(prepared, horizon_steps=HORIZON_STEPS)

    valid = labelled.dropna(subset=[TARGET_COLUMN])
    assert valid["simulation_time"].max() == DURATION - HORIZON
    assert labelled.loc[
        labelled["simulation_time"] > DURATION - HORIZON, TARGET_COLUMN
    ].isna().all()


def test_duplicate_rows_are_rejected() -> None:
    data = make_observations(duration=50)
    duplicated = pd.concat([data, data.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        prepare_observations(duplicated, duration=50, interval=INTERVAL)


def test_timestamp_gaps_are_rejected() -> None:
    data = make_observations(duration=50)
    missing = ~(
        (data["simulation_time"] == 25)
        & (data["direction"] == "north")
        & (data["run_id"] == "run_a")
    )

    with pytest.raises(ValueError, match="Incomplete"):
        prepare_observations(data[missing].copy(), duration=50, interval=INTERVAL)


def test_feature_sets_are_incremental() -> None:
    columns = [set(feature_columns(name)) for name in FEATURE_SETS]

    for earlier, later in zip(columns, columns[1:]):
        assert earlier < later
    assert len(feature_columns("F4")) == len(set(feature_columns("F4")))


def test_features_ignore_future_observations() -> None:
    prepared = prepare_observations(
        make_observations(), duration=DURATION, interval=INTERVAL
    )
    baseline = build_features(add_target(prepared, horizon_steps=HORIZON_STEPS))

    perturbed = prepared.copy()
    mask = (
        (perturbed["run_id"] == "run_a")
        & (perturbed["direction"] == "north")
        & (perturbed["simulation_time"] == 1200)
    )
    perturbed.loc[mask, "vehicle_count"] += 100
    changed = build_features(add_target(perturbed, horizon_steps=HORIZON_STEPS))

    feature_names = list(FEATURE_SET_COLUMNS["F4"])
    before = baseline[
        (baseline["run_id"] == "run_a")
        & (baseline["direction"] == "north")
        & (baseline["simulation_time"] == 1195)
    ].iloc[0][feature_names]
    after = changed[
        (changed["run_id"] == "run_a")
        & (changed["direction"] == "north")
        & (changed["simulation_time"] == 1195)
    ].iloc[0][feature_names]

    pd.testing.assert_series_equal(before, after)


def test_splits_are_chronological_with_embargo() -> None:
    dataset = build_labelled_dataset(make_observations())

    train = dataset[dataset[SPLIT_COLUMN] == "train"]
    validation = dataset[dataset[SPLIT_COLUMN] == "validation"]
    test = dataset[dataset[SPLIT_COLUMN] == "test"]

    assert len(train) == 300 * len(DIRECTIONS)
    assert len(validation) == 120 * len(DIRECTIONS)
    assert len(test) == 60 * len(DIRECTIONS)

    assert (train["simulation_time"] + HORIZON).max() < validation["simulation_time"].min()
    assert (validation["simulation_time"] + HORIZON).max() < test["simulation_time"].min()

    keys = [
        set(zip(part["run_id"], part["simulation_time"], part["direction"]))
        for part in (train, validation, test)
    ]
    assert not (keys[0] & keys[1])
    assert not (keys[1] & keys[2])
    assert not (keys[0] & keys[2])


def test_assign_splits_rejects_short_embargo() -> None:
    data = make_observations()
    prepared = prepare_observations(data, duration=DURATION, interval=INTERVAL)
    bad = dict(SPLIT_CONFIG, embargo_seconds=60)

    with pytest.raises(ValueError, match="embargo"):
        assign_splits(prepared, split_config=bad, duration=DURATION, horizon_seconds=HORIZON)


def test_build_dataset_writes_outputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    observations = make_observations()
    observations.to_csv(raw_dir / "medium_seed42.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_id": "medium_seed42",
                "scenario": "medium",
                "seed": 42,
                "status": "ok",
                "duration": DURATION,
                "sampling_interval": INTERVAL,
                "simulation_end": DURATION,
            }
        ]
    ).to_csv(raw_dir / "run_manifest.csv", index=False)

    config = {
        "data": {
            "raw_dir": str(raw_dir),
            "manifest": "run_manifest.csv",
            "processed_dir": str(tmp_path / "processed"),
            "splits_dir": "splits",
        },
        "traffic_config": str(
            Path(__file__).resolve().parents[1] / "configs" / "traffic_scenarios.yaml"
        ),
        "split": SPLIT_CONFIG,
        "features": {"default_set": "F4", "warmup_seconds": 300},
        "output": {
            "dataset": "sl_dataset.csv",
            "feature_map": "feature_columns.json",
            "audit": "data_audit.json",
        },
    }

    dataset, audit = build_dataset(
        config, raw_dir=raw_dir, processed_dir=tmp_path / "processed"
    )

    assert len(dataset) == 1920
    assert audit["split_rows"] == {"train": 1200, "validation": 480, "test": 240}
    assert (tmp_path / "processed" / "sl_dataset.csv").exists()
    assert (tmp_path / "processed" / "splits" / "train.csv").exists()
    assert (tmp_path / "processed" / "feature_columns.json").exists()
    assert (tmp_path / "processed" / "data_audit.json").exists()
