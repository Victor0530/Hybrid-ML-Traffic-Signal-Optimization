"""Reinforcement-learning evaluation entry point.

Runs a trained agent (or the fixed-time baseline) over the evaluation
scenario/seed grid from ``configs/rl.yaml`` and writes per-episode metrics
plus a mean/std summary in the shared traffic-metric definitions.

Usage:
    python -m src.reinforcement.evaluate --agent fixed
    python -m src.reinforcement.evaluate --agent <registered_agent> --checkpoint models/rl/<agent>/<run>
    python -m src.reinforcement.evaluate --agent <registered_agent> --scenarios low high --seeds 42 43

``--agent fixed`` evaluates the native 90 s programme (no checkpoint) and always
works. Learned agents must be registered in ``src/reinforcement/agents.py`` and
run greedily (epsilon = 0) so results are deterministic for a given seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.reinforcement.agents import available_agents, make_agent
from src.reinforcement.environment import TrafficEnvironment
from src.utils import load_yaml, set_global_seed

DEFAULT_RL_CONFIG = Path("configs/rl.yaml")
DEFAULT_AGENTS_CONFIG = Path("configs/rl_agents.yaml")

EVAL_COLUMNS = [
    "agent",
    "scenario",
    "seed",
    "episode",
    "total_reward",
    "steps",
    "mean_queue_total",
    "mean_waiting_total",
    "throughput_veh_per_h",
    "arrived_vehicles",
    "mean_travel_time_s",
]

SUMMARY_METRICS = (
    "total_reward",
    "mean_queue_total",
    "mean_waiting_total",
    "throughput_veh_per_h",
    "arrived_vehicles",
    "mean_travel_time_s",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, help="'fixed' or a registered agent name")
    parser.add_argument("--checkpoint", type=Path, default=None, help="directory holding the trained checkpoint")
    parser.add_argument("--config", type=Path, default=DEFAULT_RL_CONFIG)
    parser.add_argument("--agents-config", type=Path, default=DEFAULT_AGENTS_CONFIG)
    parser.add_argument("--scenarios", nargs="+", default=None, help="default: rl.yaml evaluation.scenarios")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="default: rl.yaml evaluation.seeds")
    parser.add_argument("--episodes", type=int, default=1, help="episodes per scenario/seed cell")
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--state", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/rl"))
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args(argv)


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"mean": math.nan, "std": math.nan, "n": 0}
    return {
        "mean": float(np.mean(finite)),
        # Sample std (ddof=1); undefined for n=1, reported as 0.
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "n": int(finite.size),
    }


def evaluate(
    agent_name: str,
    rl_config: dict[str, Any],
    agents_config: dict[str, Any] | None = None,
    checkpoint: str | Path | None = None,
    scenarios: list[str] | None = None,
    seeds: list[int] | None = None,
    episodes: int = 1,
    duration: int | None = None,
    state: str | None = None,
    output_dir: Path = Path("results/rl"),
    gui: bool = False,
) -> dict[str, Any]:
    """Evaluate one policy over the scenario/seed grid; return the summary."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    evaluation = dict(rl_config.get("evaluation") or {})
    scenario_list = list(scenarios or evaluation.get("scenarios") or ["medium"])
    seed_list = [int(value) for value in (seeds or evaluation.get("seeds") or [42])]

    agent: Any = None
    if agent_name != "fixed":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for learned agents")
        params = _merged_params(rl_config, agents_config or {}, agent_name, state)
        # Build once to resolve the registered class, then restore weights.
        agent = make_agent(agent_name, params)
        agent = type(agent).load(Path(checkpoint))
        # Greedy evaluation: force every exploration switch the agent exposes.
        agent.explore = False
        if hasattr(agent, "epsilon"):
            agent.epsilon = 0.0

    set_global_seed(int(seed_list[0]))
    environment = TrafficEnvironment(
        config=rl_config,
        scenario=scenario_list[0],
        seed=seed_list[0],
        gui=gui,
        duration=duration,
        fixed_time=agent_name == "fixed",
    )

    rows: list[dict[str, Any]] = []
    try:
        for scenario in scenario_list:
            for seed in seed_list:
                for episode in range(episodes):
                    observation = environment.reset(
                        scenario=scenario, seed=seed, fixed_time=agent_name == "fixed"
                    )
                    if agent is not None:
                        agent.reset()
                    total_reward = 0.0
                    steps = 0
                    while True:
                        action = None if agent is None else agent.act(observation)
                        result = environment.step(action)
                        total_reward += result.reward
                        steps += 1
                        observation = result.observation
                        if result.terminated or result.truncated:
                            break
                    metrics = dict(result.info.get("episode_metrics") or {})
                    rows.append(
                        {
                            "agent": agent_name,
                            "scenario": scenario,
                            "seed": seed,
                            "episode": episode,
                            "total_reward": round(total_reward, 4),
                            "steps": steps,
                            "mean_queue_total": metrics.get("mean_queue_total", ""),
                            "mean_waiting_total": metrics.get("mean_waiting_total", ""),
                            "throughput_veh_per_h": metrics.get("throughput_veh_per_h", ""),
                            "arrived_vehicles": metrics.get("arrived_vehicles", ""),
                            "mean_travel_time_s": metrics.get("mean_travel_time_s", ""),
                        }
                    )
    finally:
        environment.close()

    out_dir = Path(output_dir) / agent_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"eval_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVAL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "agent": agent_name,
        "scenarios": scenario_list,
        "seeds": seed_list,
        "episodes_per_cell": episodes,
        "duration": duration,
        "n_rows": len(rows),
        "metrics": {},
        "per_scenario": {},
        "csv": str(csv_path),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for metric in SUMMARY_METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric) not in ("", None)]
        summary["metrics"][metric] = _mean_std(values)
    for scenario in scenario_list:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        summary["per_scenario"][scenario] = {
            metric: _mean_std(
                [float(row[metric]) for row in scenario_rows if row.get(metric) not in ("", None)]
            )
            for metric in SUMMARY_METRICS
        }

    json_path = out_dir / f"eval_{stamp}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    summary["summary"] = str(json_path)

    print(f"{agent_name}: {len(rows)} episodes -> {csv_path}")
    for metric in ("mean_waiting_total", "mean_queue_total", "throughput_veh_per_h"):
        stats = summary["metrics"][metric]
        print(f"  {metric:>22}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['n']})")
    return summary


def _merged_params(
    rl_config: dict[str, Any],
    agents_config: dict[str, Any],
    agent_name: str,
    state: str | None,
) -> dict[str, Any]:
    from src.reinforcement.train import resolve_config

    params = resolve_config(rl_config, agents_config, agent_name)
    if state:
        params["state"] = state
    params.setdefault("state", (rl_config.get("environment") or {}).get("state", "S0"))
    return params


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.agent != "fixed" and args.agent not in available_agents():
        raise SystemExit(
            f"Unknown agent {args.agent!r}. Use 'fixed' or one of: "
            f"{', '.join(available_agents()) or '(none registered yet)'}. "
            "Register yours in src/reinforcement/agents.py."
        )
    rl_config = load_yaml(args.config)
    agents_config = load_yaml(args.agents_config) if args.agents_config.is_file() else {}
    evaluate(
        agent_name=args.agent,
        rl_config=rl_config,
        agents_config=agents_config,
        checkpoint=args.checkpoint,
        scenarios=args.scenarios,
        seeds=args.seeds,
        episodes=args.episodes,
        duration=args.duration,
        state=args.state,
        output_dir=args.output_dir,
        gui=args.gui,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
