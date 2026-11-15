"""Reference signal controllers used as baselines.

The fixed-time controller follows the same 90 s static plan as the SUMO
network. The environment can also let SUMO run that plan natively with
``fixed_time=True``; this class provides the same plan as an explicit policy
for runners that want one uniform controller interface.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.reinforcement.environment import Observation


@dataclass(frozen=True)
class PhaseTiming:
    phase: int
    duration: float


FIXED_TIME_PLAN: tuple[PhaseTiming, ...] = (
    PhaseTiming(phase=0, duration=33.0),
    PhaseTiming(phase=1, duration=3.0),
    PhaseTiming(phase=2, duration=6.0),
    PhaseTiming(phase=3, duration=3.0),
    PhaseTiming(phase=4, duration=33.0),
    PhaseTiming(phase=5, duration=3.0),
    PhaseTiming(phase=6, duration=6.0),
    PhaseTiming(phase=7, duration=3.0),
)


class FixedTimeController:
    """Map simulation time onto the network's static 90 s signal plan."""

    def __init__(self, plan: tuple[PhaseTiming, ...] = FIXED_TIME_PLAN) -> None:
        if not plan:
            raise ValueError("Fixed-time plan must not be empty")
        if any(item.duration <= 0 for item in plan):
            raise ValueError("Every fixed-time phase duration must be positive")
        self.plan = plan
        self.cycle_seconds = sum(item.duration for item in plan)

    def phase_for_time(self, simulation_time: float) -> int:
        if simulation_time < 0:
            raise ValueError("simulation_time must be non-negative")
        position = float(simulation_time) % self.cycle_seconds
        elapsed = 0.0
        for item in self.plan:
            elapsed += item.duration
            if position < elapsed:
                return item.phase
        return self.plan[-1].phase

    def reset(self) -> None:
        """Present for the common controller interface; the plan is stateless."""

    def act(self, observation: Observation) -> int:
        return self.phase_for_time(observation.simulation_time)
