"""Tests for dataset schema validation and configuration hashing."""

from __future__ import annotations

import csv
from pathlib import Path

from src.simulation.data_collector import (
    OBSERVATION_FIELDS,
    hash_config,
    stable_network_hash,
    validate_run_csv,
)
from src.simulation.scenario_generator import DIRECTIONS, load_config


def write_observation_csv(
    path: Path,
    run_id: str,
    times: tuple[int, ...] = (5, 10),
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVATION_FIELDS)
        writer.writeheader()
        for simulation_time in times:
            for direction in DIRECTIONS:
                writer.writerow(
                    {
                        "run_id": run_id,
                        "scenario": "medium",
                        "seed": 42,
                        "simulation_time": simulation_time,
                        "direction": direction,
                        "vehicle_count": 1,
                        "mean_speed": "10.000",
                        "queue_length": 0,
                        "waiting_time": "0.000",
                        "occupancy": "0.010",
                        "traffic_light_phase": 0,
                        "phase_duration": "33.0",
                        "time_in_phase": "1.0",
                        "no_vehicles": 0,
                        "sensor_missing": 0,
                        "departed_count": 0,
                        "arrived_count": 0,
                    }
                )


def test_valid_run_csv_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "medium_seed42.csv"
    write_observation_csv(path, "medium_seed42")

    valid, reason = validate_run_csv(path, "medium_seed42", duration=10, interval=5)

    assert valid is True
    assert reason == "ok"


def test_wrong_run_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "medium_seed42.csv"
    write_observation_csv(path, "medium_seed42")

    valid, reason = validate_run_csv(path, "medium_seed43", duration=10, interval=5)

    assert valid is False
    assert "run_id" in reason


def test_old_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "medium_seed42.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVATION_FIELDS[:-1])
        writer.writeheader()

    valid, reason = validate_run_csv(path, "medium_seed42", duration=10, interval=5)

    assert valid is False
    assert "schema" in reason


def test_hash_config_is_stable_and_sensitive() -> None:
    config = load_config()

    assert hash_config(config) == hash_config(load_config())

    changed = load_config()
    changed["scenarios"]["high"]["demand"]["north"] += 1
    assert hash_config(changed) != hash_config(config)


def test_stable_network_hash_ignores_generated_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "intersection.net.xml"
    template = """<?xml version="1.0" encoding="UTF-8"?>
<!-- generated on {timestamp} by Eclipse SUMO netconvert 1.28.0 -->
<net version="1.20">
    <edge id="north_in"/>
</net>
"""

    path.write_text(template.format(timestamp="2026-11-13T09:05:18"), encoding="utf-8")
    first = stable_network_hash(path)

    path.write_text(template.format(timestamp="2026-11-14T10:00:00"), encoding="utf-8")
    assert stable_network_hash(path) == first

    path.write_text(
        template.format(timestamp="2026-11-14T10:00:00").replace(
            'id="north_in"', 'id="north_in_changed"'
        ),
        encoding="utf-8",
    )
    assert stable_network_hash(path) != first
