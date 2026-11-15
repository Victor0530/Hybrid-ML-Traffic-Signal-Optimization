"""Contract tests for the RL observation, action and fixed-time interfaces."""

from __future__ import annotations

import numpy as np
import pytest

from src.reinforcement.controllers import FIXED_TIME_PLAN, FixedTimeController
from src.reinforcement.environment import (
    GREEN_PHASES,
    Observation,
    TrafficEnvironment,
    observation_size,
)


def sample_observation(phase: int = 0) -> Observation:
    return Observation(
        simulation_time=10.0,
        phase=phase,
        time_in_phase=5.0,
        phase_duration=15.0,
        counts=(4, 3, 2, 1),
        queues=(2, 1, 0, 1),
        speeds=(10.0, 11.0, 12.0, 13.0),
        waiting=(20.0, 10.0, 0.0, 5.0),
    )


def test_observation_vectors_have_documented_sizes() -> None:
    observation = sample_observation()

    assert observation_size("S0") == 20
    assert observation_size("S1") == 21
    assert observation_size("S2") == 25
    assert observation.as_vector("S0").shape == (20,)
    assert observation.as_vector("S1").shape == (21,)
    assert observation.as_vector("S2", predictions=[1, 2, 3, 4]).shape == (25,)


def test_s2_requires_one_prediction_per_direction() -> None:
    observation = sample_observation()

    with pytest.raises(ValueError):
        observation.as_vector("S2")
    with pytest.raises(ValueError):
        observation.as_vector("S2", predictions=[1, 2, 3])


def test_phase_one_hot_is_in_the_expected_position() -> None:
    vector = sample_observation(phase=2).as_vector("S0")

    assert np.array_equal(vector[-8:], np.array([0, 0, 1, 0, 0, 0, 0, 0]))


def test_fixed_time_controller_follows_the_90_second_plan() -> None:
    controller = FixedTimeController()
    assert controller.cycle_seconds == pytest.approx(90.0)
    assert sum(item.duration for item in FIXED_TIME_PLAN) == pytest.approx(90.0)

    assert controller.phase_for_time(0) == 0
    assert controller.phase_for_time(32.9) == 0
    assert controller.phase_for_time(33.0) == 1
    assert controller.phase_for_time(36.0) == 2
    assert controller.phase_for_time(42.0) == 3
    assert controller.phase_for_time(45.0) == 4
    assert controller.phase_for_time(78.0) == 5
    assert controller.phase_for_time(81.0) == 6
    assert controller.phase_for_time(87.0) == 7
    assert controller.phase_for_time(90.0) == 0


def test_environment_exposes_a_small_safe_action_space() -> None:
    environment = TrafficEnvironment(scenario="medium", seed=42)

    try:
        assert environment.green_phases == GREEN_PHASES
        assert environment.green_to_yellow == {0: 1, 2: 3, 4: 5, 6: 7}
        assert environment.decision_interval == 15.0
        assert environment.yellow_seconds == 3.0
        assert environment.validate_action(4) == 4
        with pytest.raises(ValueError):
            environment.validate_action(1)
    finally:
        environment.close()


def test_environment_rejects_unknown_scenarios() -> None:
    with pytest.raises(ValueError):
        TrafficEnvironment(scenario="does_not_exist")
