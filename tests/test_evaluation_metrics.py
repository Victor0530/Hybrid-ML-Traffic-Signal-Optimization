"""Tests for the shared regression and traffic metrics."""

from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import aggregate_traffic_metrics, regression_metrics


def test_regression_metrics_have_expected_values() -> None:
    metrics = regression_metrics([1, 2, 3], [1, 2, 4])

    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["rmse"] == pytest.approx(math.sqrt(1 / 3))
    assert metrics["r2"] == pytest.approx(0.5)
    assert metrics["mape"] == pytest.approx((1 / 3) / 3 * 100)


def test_regression_metrics_ignore_zero_targets_for_mape() -> None:
    metrics = regression_metrics([0, 2], [1, 2])

    assert metrics["mape"] == 0.0


def test_regression_metrics_return_nan_r2_for_constant_target() -> None:
    metrics = regression_metrics([2, 2, 2], [2, 2, 2])

    assert math.isnan(metrics["r2"])


def test_regression_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        regression_metrics([1, 2], [1])


def test_aggregate_traffic_metrics_uses_one_definition() -> None:
    records = [
        {"queue_length": [2, 4, 1, 3], "waiting_time": [10, 20, 5, 15], "arrived": 2},
        {"queue_length": [1, 1, 1, 1], "waiting_time": [5, 5, 5, 5], "arrived": 3},
    ]

    metrics = aggregate_traffic_metrics(
        records,
        duration_seconds=60,
        travel_times=[10, 20, 30],
        approaches=4,
    )

    assert metrics["mean_queue_total"] == pytest.approx(7.0)
    assert metrics["mean_queue_per_approach"] == pytest.approx(1.75)
    assert metrics["max_queue_total"] == 10.0
    assert metrics["mean_waiting_total"] == pytest.approx(35.0)
    assert metrics["mean_waiting_per_approach"] == pytest.approx(8.75)
    assert metrics["throughput_veh_per_h"] == pytest.approx(300.0)
    assert metrics["arrived_vehicles"] == 5
    assert metrics["completed_vehicles"] == 3
    assert metrics["mean_travel_time_s"] == pytest.approx(20.0)


def test_aggregate_traffic_metrics_handles_no_completed_trips() -> None:
    metrics = aggregate_traffic_metrics(
        [{"queue_length": 2, "waiting_time": 4, "arrived": 0}],
        duration_seconds=30,
        travel_times=[],
    )

    assert metrics["completed_vehicles"] == 0
    assert math.isnan(metrics["mean_travel_time_s"])
