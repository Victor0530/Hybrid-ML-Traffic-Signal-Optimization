"""Tests for deterministic scenario route generation."""

from __future__ import annotations

import pytest

from src.simulation.scenario_generator import (
    DIRECTIONS,
    MOVEMENTS,
    build_route_xml,
    get_demand_phases,
    load_config,
    write_scenario_routes,
)


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


def test_all_scenarios_have_demand(config: dict) -> None:
    assert set(config["scenarios"]) == {
        "low",
        "medium",
        "high",
        "uneven",
        "changing",
    }
    for name, scenario in config["scenarios"].items():
        assert "demand" in scenario or "phases" in scenario, name


def test_medium_has_twelve_routes_and_flows(config: dict) -> None:
    xml = build_route_xml("medium", config)

    assert xml.count('<route id="route_') == len(DIRECTIONS) * len(MOVEMENTS)
    assert xml.count("<flow ") == len(DIRECTIONS) * len(MOVEMENTS)
    # 600 veh/h per approach x 0.70 straight = 420 veh/h
    assert 'vehsPerHour="420.0"' in xml
    assert 'vehsPerHour="90.0"' in xml


def test_changing_scenario_has_three_demand_phases(config: dict) -> None:
    phases = get_demand_phases(
        config["scenarios"]["changing"], config["simulation"]["duration"]
    )

    assert len(phases) == 3
    assert [(phase["begin"], phase["end"]) for phase in phases] == [
        (0, 1800),
        (1800, 2700),
        (2700, 3600),
    ]
    xml = build_route_xml("changing", config)
    assert xml.count("<flow ") == 3 * len(DIRECTIONS) * len(MOVEMENTS)


def test_high_scenario_is_calibrated_at_700(config: dict) -> None:
    demand = config["scenarios"]["high"]["demand"]
    assert all(rate == 700 for rate in demand.values())


def test_written_route_files_are_byte_identical(config: dict, tmp_path) -> None:
    first = write_scenario_routes("changing", config, tmp_path / "first")
    second = write_scenario_routes("changing", config, tmp_path / "second")

    assert first.read_bytes() == second.read_bytes()
