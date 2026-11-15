"""Tests for the supervised-learning EDA helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.explore_sl_dataset import (
    autocorrelation_table,
    build_overview,
    correlation_table,
    distribution_summary,
    safe_correlation,
    split_shift_table,
)


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for run_id, offset in (("run_a", 0), ("run_b", 10)):
        for index, time in enumerate(range(0, 25, 5)):
            for direction in ("north", "south", "east", "west"):
                count = offset + index + (1 if direction == "north" else 0)
                rows.append(
                    {
                        "run_id": run_id,
                        "scenario": "test",
                        "seed": 42,
                        "simulation_time": time,
                        "direction": direction,
                        "split": "train" if time < 10 else "validation",
                        "vehicle_count": count,
                        "target_vehicle_count": count + 1,
                    }
                )
    return pd.DataFrame(rows)


def test_distribution_summary_groups_rows() -> None:
    data = synthetic_frame()
    summary = distribution_summary(data, ["split"])

    assert list(summary.columns[:3]) == ["split", "rows", "mean"]
    assert set(summary["split"]) == {"train", "validation"}
    assert summary["rows"].sum() == len(data)
    train_mean = data.loc[data["split"] == "train", "target_vehicle_count"].mean()
    assert summary.loc[summary["split"] == "train", "mean"].iloc[0] == pytest.approx(
        train_mean
    )


def test_correlation_is_nan_for_constant_series() -> None:
    constant = pd.Series([1.0, 1.0, 1.0, 1.0])
    varying = pd.Series([1.0, 2.0, 3.0, 4.0])

    assert np.isnan(safe_correlation(constant, varying))
    assert safe_correlation(varying, varying) == pytest.approx(1.0)


def test_correlation_table_orders_by_absolute_correlation() -> None:
    data = synthetic_frame()
    data["strong"] = data["target_vehicle_count"] * 2
    data["weak"] = data["target_vehicle_count"] * -0.1 + (np.arange(len(data)) % 3)
    data["constant"] = 1.0

    table = correlation_table(data, ["weak", "strong", "constant"])
    assert table.loc[0, "feature"] == "strong"
    assert table.loc[table["feature"] == "constant", "corr_all"].isna().all()
    assert table.loc[table["feature"] == "strong", "corr_all"].iloc[0] == pytest.approx(
        1.0
    )


def test_split_shift_is_zero_for_identical_distributions() -> None:
    rows = []
    for split in ("train", "validation", "test"):
        for index in range(8):
            rows.append(
                {
                    "split": split,
                    "target_vehicle_count": float(index),
                }
            )
    data = pd.DataFrame(rows)

    table = split_shift_table(data, [])
    target_row = table.loc[table["column"] == "target_vehicle_count"].iloc[0]
    assert target_row["smd_train_validation"] == pytest.approx(0.0)
    assert target_row["smd_train_test"] == pytest.approx(0.0)


def test_autocorrelation_never_crosses_run_boundaries() -> None:
    data = synthetic_frame()
    table = autocorrelation_table(data, lags=(1,))
    all_row = table.loc[table["scope"] == "all"].iloc[0]

    # Two runs x four directions x 5 timestamps = 40 rows; lag 1 leaves 4 pairs
    # per group, so 2 x 4 x 4 = 32 valid pairs. A cross-run shift would add 8.
    assert int(all_row["pairs"]) == 32
    assert -1.0 <= all_row["autocorrelation"] <= 1.0


def test_build_overview_reports_clean_dataset_facts() -> None:
    data = synthetic_frame()
    feature_map = {
        "F0": ["vehicle_count"],
        "F1": ["vehicle_count"],
        "F2": [],
        "F3": [],
        "F4": [],
    }

    overview = build_overview(data, feature_map)

    assert overview["rows"] == len(data)
    assert overview["runs"] == 2
    assert overview["missing_cells"] == 0
    assert overview["simulation_time"]["step_seconds"] == 5
    assert overview["feature_counts"]["F0"] == 1
