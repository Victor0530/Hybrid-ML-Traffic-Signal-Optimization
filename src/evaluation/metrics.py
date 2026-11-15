"""Shared metrics for SL predictions and RL traffic-control episodes.

The important contract is that every experiment uses the same definitions:
queue and waiting time are summed over the four approaches at each simulation
step, throughput is completed vehicles per hour, and travel time is the mean
time from departure to arrival for vehicles that completed their trip.

Units:
    queue_length        vehicles (sum over approaches)
    waiting_time        seconds (sum over approaches)
    throughput          vehicles per hour
    travel_time         seconds per completed vehicle
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


def _finite_array(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def regression_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> dict[str, float]:
    """Return MAE, RMSE, R² and MAPE for regression predictions.

    MAPE ignores observations whose true value is zero; it is reported as a
    percentage. R² is NaN when the target has no variance.
    """
    truth = _finite_array(y_true, "y_true")
    prediction = _finite_array(y_pred, "y_pred")
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same length")

    error = prediction - truth
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))

    total_sum_squares = float(np.sum((truth - truth.mean()) ** 2))
    residual_sum_squares = float(np.sum(error**2))
    r2 = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0.0
        else float("nan")
    )

    non_zero = truth != 0.0
    mape = (
        float(np.mean(np.abs(error[non_zero] / truth[non_zero])) * 100.0)
        if np.any(non_zero)
        else float("nan")
    )

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def _step_total(
    records: Sequence[Mapping[str, Any]],
    key: str,
    approaches: int,
) -> list[float]:
    """Sum a per-approach metric at each step, returning one total per step."""
    totals: list[float] = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size == 1:
            totals.append(float(array[0]))
        elif array.size == approaches:
            totals.append(float(array.sum()))
        else:
            raise ValueError(
                f"{key!r} must be a scalar or have {approaches} approach values"
            )
    return totals


def aggregate_traffic_metrics(
    step_records: Sequence[Mapping[str, Any]],
    duration_seconds: float,
    travel_times: Sequence[float] | None = None,
    approaches: int = 4,
) -> dict[str, float | int]:
    """Aggregate one episode or run into the shared traffic metrics.

    ``step_records`` is a sequence of per-decision-step mappings. Each mapping
    may contain ``queue_length`` and ``waiting_time`` as either one scalar or a
    sequence with one value per approach, plus an integer ``arrived`` count.
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if approaches <= 0:
        raise ValueError("approaches must be positive")

    queue_totals = _step_total(step_records, "queue_length", approaches)
    waiting_totals = _step_total(step_records, "waiting_time", approaches)
    arrived = sum(int(record.get("arrived", 0)) for record in step_records)

    mean_queue = float(np.mean(queue_totals)) if queue_totals else float("nan")
    mean_waiting = (
        float(np.mean(waiting_totals)) if waiting_totals else float("nan")
    )
    travel = (
        np.asarray(list(travel_times), dtype=float).reshape(-1)
        if travel_times is not None
        else np.asarray([], dtype=float)
    )

    return {
        "mean_queue_total": mean_queue,
        "max_queue_total": (
            float(np.max(queue_totals)) if queue_totals else float("nan")
        ),
        "mean_queue_per_approach": mean_queue / approaches,
        "mean_waiting_total": mean_waiting,
        "max_waiting_total": (
            float(np.max(waiting_totals)) if waiting_totals else float("nan")
        ),
        "mean_waiting_per_approach": mean_waiting / approaches,
        "throughput_veh_per_h": arrived / duration_seconds * 3600.0,
        "arrived_vehicles": arrived,
        "completed_vehicles": int(travel.size),
        "mean_travel_time_s": (
            float(np.mean(travel)) if travel.size else float("nan")
        ),
        "episode_duration_s": float(duration_seconds),
    }
