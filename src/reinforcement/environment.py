"""TraCI reinforcement-learning environment contract.

The wrapper exposes a small, stable interface for every RL algorithm:

    env = TrafficEnvironment(scenario="medium", seed=42)
    observation = env.reset()
    result = env.step(action)       # action is a green phase index
    env.close()

Actions select one of the four green phases. Yellow/clearance phases are
inserted by the environment, never chosen by the agent. Passing ``None`` keeps
the current valid phase. With ``fixed_time=True`` the SUMO static programme
runs normally and the action is ignored, which is the fixed-time baseline.

Safe transitions follow the native programme's clearance rule: whenever a
through green (0, 4) is left for a non-adjacent green, the environment runs
the paired yellow, the protected-left flush phase and its yellow before the
target green (0 -> 1 -> 2 -> 3 -> 4). This guarantees that permissive
left-turners already inside the junction get a protected phase to exit
before cross traffic starts, which prevents junction gridlock.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import traci
import yaml

from src.evaluation.metrics import aggregate_traffic_metrics
from src.simulation.scenario_generator import (
    DIRECTIONS,
    load_config as load_traffic_config,
    write_scenario_routes,
)
from src.simulation.sumo_binary import resolve_sumo_binary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rl.yaml"
DEFAULT_TLS_ID = "C"
GREEN_PHASES = (0, 2, 4, 6)
YELLOW_BY_GREEN = {0: 1, 2: 3, 4: 5, 6: 7}
YELLOW_TO_GREEN = {yellow: green for green, yellow in YELLOW_BY_GREEN.items()}
STATE_SIZES = {"S0": 20, "S1": 21, "S2": 25}


def clearance_path(current: int, target: int, through_greens: frozenset[int]) -> list[int]:
    """Phases to run between two greens so no movement loses its clearance.

    Mirrors the native programme (0->1->2->3->4->5->6->7):

    - leaving a through green for a non-adjacent green runs the paired
      yellow, the protected-left flush and its yellow (0 -> 4 gives
      [1, 2, 3]) so trapped left-turners exit before cross traffic starts;
    - leaving a through green for its own left phase runs only the paired
      yellow (0 -> 2 gives [1]);
    - leaving a left green runs only its yellow (2 -> 0 gives [3]).
    """
    if current == target:
        return []
    if current not in YELLOW_BY_GREEN:
        raise ValueError(f"{current!r} is not a green phase")
    greens = sorted(YELLOW_BY_GREEN)  # [0, 2, 4, 6]
    next_green = greens[(greens.index(current) + 1) % len(greens)]
    path = [YELLOW_BY_GREEN[current]]
    if current in through_greens and target != next_green:
        path.extend([next_green, YELLOW_BY_GREEN[next_green]])
    return path


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_rl_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with _resolve_path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def observation_size(state: str) -> int:
    try:
        return STATE_SIZES[state]
    except KeyError as exc:
        raise ValueError(
            f"Unknown state {state!r}; choose from {tuple(STATE_SIZES)}"
        ) from exc


@dataclass(frozen=True)
class Observation:
    """Current traffic and signal state for one approach-ordered snapshot."""

    simulation_time: float
    phase: int
    time_in_phase: float
    phase_duration: float
    counts: tuple[float, ...]
    queues: tuple[float, ...]
    speeds: tuple[float, ...]
    waiting: tuple[float, ...]

    def as_vector(
        self,
        state: str = "S0",
        predictions: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the fixed-order numeric state used by RL agents.

        S0: counts, queues, speeds, phase one-hot (20 values).
        S1: S0 + time in phase (21 values).
        S2: S1 + four SL-predicted future counts (25 values).
        """
        if state not in STATE_SIZES:
            raise ValueError(f"Unknown state {state!r}; choose from {tuple(STATE_SIZES)}")
        if self.phase < 0 or self.phase >= 8:
            raise ValueError(f"Phase must be in [0, 8), got {self.phase}")

        phase_one_hot = np.zeros(8, dtype=float)
        phase_one_hot[self.phase] = 1.0
        base = np.concatenate(
            [
                np.asarray(self.counts, dtype=float),
                np.asarray(self.queues, dtype=float),
                np.asarray(self.speeds, dtype=float),
                phase_one_hot,
            ]
        )
        if state == "S0":
            return base

        with_time = np.concatenate([base, [float(self.time_in_phase)]])
        if state == "S1":
            return with_time

        if predictions is None:
            raise ValueError("State S2 requires four SL predictions")
        predicted = np.asarray(predictions, dtype=float).reshape(-1)
        if predicted.size != len(DIRECTIONS):
            raise ValueError(
                f"State S2 needs {len(DIRECTIONS)} predictions, "
                f"got {predicted.size}"
            )
        return np.concatenate([with_time, predicted])

    @property
    def total_queue(self) -> float:
        return float(sum(self.queues))

    @property
    def total_waiting(self) -> float:
        return float(sum(self.waiting))


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class Policy(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: Observation) -> int | None: ...


class TrafficEnvironment:
    """One SUMO intersection with a safe phase-selection action space."""

    def __init__(
        self,
        config: dict[str, Any] | str | Path | None = None,
        scenario: str = "medium",
        seed: int = 42,
        gui: bool = False,
        fixed_time: bool = False,
        duration: int | None = None,
        tls_id: str | None = None,
    ) -> None:
        cfg = load_rl_config(config or DEFAULT_CONFIG) if not isinstance(config, dict) else config
        self.config = cfg
        self.environment_config = cfg["environment"]
        self.reward_config = cfg["reward"]
        self.action_config = cfg["actions"]

        self.traffic_config_path = _resolve_path(
            self.environment_config["traffic_config"]
        )
        self.traffic_config = load_traffic_config(self.traffic_config_path)
        self.network_config = _resolve_path(self.environment_config["network_config"])
        self.tls_id = tls_id or str(
            self.environment_config.get("tls_id", DEFAULT_TLS_ID)
        )
        self.decision_interval = float(
            self.environment_config.get("decision_interval", 15)
        )
        self.min_green = float(self.environment_config.get("min_green", 5))
        self.max_green = float(self.environment_config.get("max_green", 60))
        self.yellow_seconds = float(
            self.environment_config.get("yellow_seconds", 3)
        )
        self.left_clearance_seconds = float(
            self.environment_config.get("left_clearance_seconds", 6)
        )
        if self.decision_interval <= 0:
            raise ValueError("decision_interval must be positive")
        if not 0 <= self.min_green <= self.max_green:
            raise ValueError("min_green must be between 0 and max_green")
        if self.yellow_seconds <= 0:
            raise ValueError("yellow_seconds must be positive")
        if self.left_clearance_seconds <= 0:
            raise ValueError("left_clearance_seconds must be positive")

        self.green_phases = tuple(
            int(phase) for phase in self.action_config["green_phases"]
        )
        self.yellow_phases = tuple(
            int(phase) for phase in self.action_config["yellow_phases"]
        )
        if len(self.green_phases) != len(self.yellow_phases):
            raise ValueError("green_phases and yellow_phases must have equal length")
        if len(set(self.green_phases)) != len(self.green_phases):
            raise ValueError("green_phases must be unique")
        if set(self.green_phases) & set(self.yellow_phases):
            raise ValueError("Green and yellow phase sets must not overlap")
        self.green_to_yellow = dict(zip(self.green_phases, self.yellow_phases))
        self.yellow_to_green = {
            yellow: green for green, yellow in self.green_to_yellow.items()
        }
        # Through greens (0, 4) must flush protected lefts before cross
        # traffic starts; derive them from the configured action names.
        names = [str(name) for name in self.action_config.get("names") or []]
        if names and len(names) == len(self.green_phases):
            self.through_green_phases = frozenset(
                green
                for green, name in zip(self.green_phases, names, strict=True)
                if "through" in name.lower()
            )
        else:
            self.through_green_phases = frozenset(self.green_phases[::2])

        if scenario not in self.traffic_config["scenarios"]:
            raise ValueError(
                f"Unknown scenario {scenario!r}; choose from "
                f"{tuple(self.traffic_config['scenarios'])}"
            )
        self.scenario = scenario
        self.seed = int(seed)
        self.gui = bool(gui)
        self.fixed_time = bool(fixed_time)
        self.duration = int(
            duration or self.traffic_config["simulation"]["duration"]
        )
        if self.duration <= 0:
            raise ValueError("duration must be positive")

        self._started = False
        self._phase_started_at = 0.0
        self._phase_duration = float("inf")
        self._last_time = 0.0
        self._step_records: list[dict[str, Any]] = []
        self._departed_at: dict[str, float] = {}
        self._travel_times: list[float] = []
        self._departed = 0
        self._arrived = 0
        self._teleports = 0
        self._episode_done = False
        self._pending_target: int | None = None
        # Unique route file per environment instance so parallel SUMO
        # processes never write the same path concurrently.
        self._route_suffix = f"_{os.getpid()}_{id(self)}"

    # ------------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------------
    def validate_action(self, action: int) -> int:
        """Validate an agent action against the configured green phases."""
        return self._validate_action(action)

    def reset(
        self,
        scenario: str | None = None,
        seed: int | None = None,
        fixed_time: bool | None = None,
    ) -> Observation:
        """Start a new SUMO episode and return the first observation."""
        self.close()
        if scenario is not None:
            if scenario not in self.traffic_config["scenarios"]:
                raise ValueError(f"Unknown scenario {scenario!r}")
            self.scenario = scenario
        if seed is not None:
            self.seed = int(seed)
        if fixed_time is not None:
            self.fixed_time = bool(fixed_time)

        route_path = write_scenario_routes(
            self.scenario, self.traffic_config, suffix=self._route_suffix
        )
        binary = resolve_sumo_binary("sumo-gui" if self.gui else "sumo")
        traci.start(
            [
                binary,
                "-c",
                str(self.network_config),
                "--route-files",
                str(route_path),
                "--seed",
                str(self.seed),
                "--end",
                str(self.duration),
                "--no-step-log",
                "true",
            ]
        )
        self._started = True
        now = float(traci.simulation.getTime())
        self._last_time = now
        self._phase_started_at = now
        self._phase_duration = float("inf")
        self._step_records = []
        self._departed_at = {}
        self._travel_times = []
        self._departed = 0
        self._arrived = 0
        self._teleports = 0
        self._episode_done = False
        self._pending_target = None

        current_phase = int(traci.trafficlight.getPhase(self.tls_id))
        if self.fixed_time:
            self._phase_duration = float(
                traci.trafficlight.getPhaseDuration(self.tls_id)
            )
        else:
            self._set_phase(current_phase, now)
        return self._observe()

    def step(self, action: int | None = None) -> StepResult:
        """Apply one action and advance by one decision interval."""
        if not self._started:
            raise RuntimeError("Call reset() before step()")
        if self._episode_done:
            raise RuntimeError("Episode is already finished")

        before_arrived = self._arrived
        deadline = min(
            self._last_time + self.decision_interval, float(self.duration)
        )
        if not self.fixed_time:
            target = self._resolve_action(action)
            self._transition_to(target, deadline)
        self._advance_to(deadline)

        observation = self._observe()
        reward = -(
            float(self.reward_config.get("alpha", 1.0)) * observation.total_queue
            + float(self.reward_config.get("beta", 0.1))
            * observation.total_waiting
        )
        terminated = self._last_time >= self.duration - 1e-9
        if terminated:
            self._episode_done = True

        record = {
            "queue_length": observation.queues,
            "waiting_time": observation.waiting,
            "arrived": self._arrived - before_arrived,
        }
        self._step_records.append(record)
        info: dict[str, Any] = {
            "simulation_time": observation.simulation_time,
            "phase": observation.phase,
            "time_in_phase": observation.time_in_phase,
            "queue_total": observation.total_queue,
            "waiting_total": observation.total_waiting,
            "arrived_total": self._arrived,
            "departed_total": self._departed,
            "teleports": self._teleports,
            "throughput_veh_per_h": (
                self._arrived / max(observation.simulation_time, 1e-9) * 3600.0
            ),
        }
        if terminated:
            info["episode_metrics"] = aggregate_traffic_metrics(
                self._step_records,
                duration_seconds=float(self.duration),
                travel_times=self._travel_times,
                approaches=len(DIRECTIONS),
            )
        return StepResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=False,
            info=info,
        )

    def close(self) -> None:
        """Close the TraCI connection if one is open."""
        if self._started:
            try:
                traci.close()
            except Exception:  # noqa: BLE001 - closing must not mask results
                pass
            self._started = False

    def run_episode(
        self,
        policy: Policy | None = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Run one full episode and return its aggregate metrics."""
        observation = self.reset()
        if policy is not None:
            policy.reset()
        total_reward = 0.0
        steps = 0
        result: StepResult | None = None
        while True:
            action = None if self.fixed_time or policy is None else policy.act(
                observation
            )
            result = self.step(action)
            total_reward += result.reward
            steps += 1
            observation = result.observation
            if result.terminated or result.truncated:
                break
            if max_steps is not None and steps >= max_steps:
                break
        metrics = dict(result.info.get("episode_metrics", {})) if result else {}
        metrics["total_reward"] = total_reward
        metrics["steps"] = steps
        return metrics

    def __enter__(self) -> "TrafficEnvironment":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal simulation helpers
    # ------------------------------------------------------------------
    def _validate_action(self, action: int) -> int:
        value = int(action)
        if value not in self.green_phases:
            raise ValueError(
                f"Invalid action {action!r}; choose a green phase from "
                f"{self.green_phases}"
            )
        return value

    def _resolve_action(self, action: int | None) -> int:
        current = int(traci.trafficlight.getPhase(self.tls_id))
        now = float(traci.simulation.getTime())

        if action is None:
            if self._pending_target is not None:
                # Resume a clearance walk that the deadline interrupted.
                target = self._pending_target
            elif current in self.green_phases:
                target = current
            elif current in self.yellow_to_green:
                target = self.yellow_to_green[current]
            else:
                target = self.green_phases[0]
        else:
            target = self._validate_action(action)

        if current in self.green_phases and self._pending_target is None:
            elapsed = now - self._phase_started_at
            if elapsed < self.min_green:
                return current
            if elapsed >= self.max_green:
                index = self.green_phases.index(current)
                return self.green_phases[(index + 1) % len(self.green_phases)]
        return target

    def _set_phase(self, phase: int, started_at: float) -> None:
        traci.trafficlight.setPhase(self.tls_id, int(phase))
        traci.trafficlight.setPhaseDuration(self.tls_id, 9999.0)
        self._phase_started_at = started_at
        self._phase_duration = 9999.0

    def _advance_to(self, target_time: float) -> None:
        while traci.simulation.getTime() + 1e-9 < target_time:
            traci.simulationStep()
            if not self.fixed_time:
                traci.trafficlight.setPhaseDuration(self.tls_id, 9999.0)
            self._track_vehicles()
            self._last_time = float(traci.simulation.getTime())
            if self._last_time >= self.duration - 1e-9:
                break

    def clearance_path(self, target: int) -> list[int]:
        """Phases to run from the current phase before reaching ``target``.

        See :func:`clearance_path`; through greens flush protected lefts so
        permissive left-turners can leave the junction before cross traffic.
        """
        current = int(traci.trafficlight.getPhase(self.tls_id))
        if current in self.yellow_to_green:
            # Mid-clearance: the finished yellow is already accounted for.
            current = self.yellow_to_green[current]
            path = clearance_path(current, target, self.through_green_phases)
            return path[1:] if path else path
        return clearance_path(current, target, self.through_green_phases)

    def _transition_to(self, target: int, deadline: float) -> None:
        current = int(traci.trafficlight.getPhase(self.tls_id))
        if current == target:
            self._pending_target = None
            return
        self._pending_target = target

        if current in self.yellow_to_green:
            # Finish the interrupted yellow, then resume the walk.
            yellow_end = self._phase_started_at + self.yellow_seconds
            self._advance_to(min(yellow_end, deadline))
            if traci.simulation.getTime() >= deadline - 1e-9:
                return

        for phase in self.clearance_path(target):
            start = float(traci.simulation.getTime())
            if start >= deadline - 1e-9:
                return
            duration = (
                self.left_clearance_seconds
                if phase in self.green_phases
                else self.yellow_seconds
            )
            self._set_phase(phase, start)
            self._advance_to(min(start + duration, deadline))
            if traci.simulation.getTime() >= deadline - 1e-9:
                return  # resume the remaining path on the next decision step

        now = float(traci.simulation.getTime())
        if now < deadline - 1e-9:
            self._set_phase(target, now)
            self._pending_target = None

    def _track_vehicles(self) -> None:
        now = float(traci.simulation.getTime())
        for vehicle_id in traci.simulation.getDepartedIDList():
            if vehicle_id not in self._departed_at:
                self._departed_at[vehicle_id] = now
                self._departed += 1
        for vehicle_id in traci.simulation.getArrivedIDList():
            self._arrived += 1
            started = self._departed_at.pop(vehicle_id, None)
            if started is not None:
                self._travel_times.append(now - started)
        for vehicle_id in traci.simulation.getStartingTeleportIDList():
            self._teleports += 1
            self._departed_at.pop(vehicle_id, None)

    def _observe(self) -> Observation:
        now = float(traci.simulation.getTime())
        phase = int(traci.trafficlight.getPhase(self.tls_id))
        if self.fixed_time:
            duration = float(traci.trafficlight.getPhaseDuration(self.tls_id))
            next_switch = float(traci.trafficlight.getNextSwitch(self.tls_id))
            elapsed = max(0.0, duration - max(0.0, next_switch - now))
        else:
            duration = self._phase_duration
            elapsed = max(0.0, now - self._phase_started_at)

        counts: list[float] = []
        queues: list[float] = []
        speeds: list[float] = []
        waiting: list[float] = []
        for direction in DIRECTIONS:
            edge = f"{direction}_in"
            counts.append(float(traci.edge.getLastStepVehicleNumber(edge)))
            queues.append(float(traci.edge.getLastStepHaltingNumber(edge)))
            speeds.append(max(0.0, float(traci.edge.getLastStepMeanSpeed(edge))))
            waiting.append(float(traci.edge.getWaitingTime(edge)))

        return Observation(
            simulation_time=now,
            phase=phase,
            time_in_phase=elapsed,
            phase_duration=duration,
            counts=tuple(counts),
            queues=tuple(queues),
            speeds=tuple(speeds),
            waiting=tuple(waiting),
        )


class _AlternatePolicy:
    """Small smoke-test policy that exercises safe phase transitions."""

    def __init__(self, every: int = 2) -> None:
        self.every = max(1, int(every))
        self.step_index = 0

    def reset(self) -> None:
        self.step_index = 0

    def act(self, observation: Observation) -> int:
        phase = GREEN_PHASES[
            (self.step_index // self.every) % len(GREEN_PHASES)
        ]
        self.step_index += 1
        return phase


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--policy",
        choices=("fixed", "alternate"),
        default="fixed",
        help="fixed uses the native static programme; alternate switches phases",
    )
    args = parser.parse_args()

    environment = TrafficEnvironment(
        config=args.config,
        scenario=args.scenario,
        seed=args.seed,
        gui=args.gui,
        fixed_time=args.policy == "fixed",
        duration=args.duration,
    )
    policy = None if args.policy == "fixed" else _AlternatePolicy()
    try:
        metrics = environment.run_episode(policy)
    finally:
        environment.close()
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
