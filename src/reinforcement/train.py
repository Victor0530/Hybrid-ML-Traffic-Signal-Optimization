"""Reinforcement-learning training entry point (lanes A and D).

Trains one registered agent on the frozen traffic environment, logs one row
of episode metrics per episode and writes a checkpoint that
``src.reinforcement.evaluate`` can load.

Usage:
    python -m src.reinforcement.train --agent <registered_agent> --scenario medium
    python -m src.reinforcement.train --agent <registered_agent> --episodes 200 --seed 42
    python -m src.reinforcement.train --agent <registered_agent> --duration 600   # smoke

No agent ships with the project: register your own in
``src/reinforcement/agents.py`` first (see that module's docstring).

Configuration precedence (later wins):
    configs/rl.yaml ``training`` block  ->  configs/rl_agents.yaml
    ``training`` block  ->  configs/rl_agents.yaml ``<agent>`` block  ->  CLI.

The environment itself always uses the frozen ``configs/rl.yaml``
environment/action/reward blocks.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.reinforcement.agents import available_agents, make_agent
from src.reinforcement.environment import TrafficEnvironment
from src.utils import deep_merge, load_yaml, set_global_seed

DEFAULT_RL_CONFIG = Path("configs/rl.yaml")
DEFAULT_AGENTS_CONFIG = Path("configs/rl_agents.yaml")

LOG_COLUMNS = [
    "episode",
    "seed",
    "scenario",
    "total_reward",
    "steps",
    "mean_queue_total",
    "mean_waiting_total",
    "throughput_veh_per_h",
    "arrived_vehicles",
    "mean_travel_time_s",
    "epsilon",
    "wall_seconds",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, help="registered agent name; see src/reinforcement/agents.py")
    parser.add_argument("--config", type=Path, default=DEFAULT_RL_CONFIG)
    parser.add_argument("--agents-config", type=Path, default=DEFAULT_AGENTS_CONFIG)
    parser.add_argument("--scenario", default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=None, help="default: rl.yaml training.episodes")
    parser.add_argument("--duration", type=int, default=None, help="sim-seconds per episode; default: traffic config")
    parser.add_argument("--state", default=None, help="S0/S1/S2 override (S2 needs SL predictions; use S0/S1 here)")
    parser.add_argument("--save-every", type=int, default=None, help="checkpoint interval in episodes; 0 = only at the end")
    parser.add_argument("--output-dir", type=Path, default=Path("results/rl"))
    parser.add_argument("--save-dir", type=Path, default=Path("models/rl"))
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args(argv)


def resolve_config(
    rl_config: dict[str, Any],
    agents_config: dict[str, Any],
    agent_name: str,
) -> dict[str, Any]:
    """Merge shared training settings with the per-agent hyper-parameters."""
    merged = deep_merge(
        dict(rl_config.get("training") or {}),
        dict(agents_config.get("training") or {}),
    )
    merged = deep_merge(merged, dict(agents_config.get(agent_name) or {}))
    return merged


def train(
    agent_name: str,
    rl_config: dict[str, Any],
    agents_config: dict[str, Any],
    scenario: str = "medium",
    seed: int = 42,
    episodes: int | None = None,
    duration: int | None = None,
    state: str | None = None,
    save_every: int | None = None,
    output_dir: Path = Path("results/rl"),
    save_dir: Path = Path("models/rl"),
    gui: bool = False,
) -> dict[str, Any]:
    """Train one agent and return a summary dict with paths and totals."""
    set_global_seed(seed)
    params = resolve_config(rl_config, agents_config, agent_name)
    if state:
        params["state"] = state
    params.setdefault("state", (rl_config.get("environment") or {}).get("state", "S0"))
    params["seed"] = seed

    n_episodes = int(episodes if episodes is not None else params.get("episodes", 200))
    if n_episodes <= 0:
        raise ValueError("episodes must be positive")
    interval = int(save_every if save_every is not None else params.get("save_every", 0) or 0)
    if interval < 0:
        raise ValueError("save_every must be non-negative")

    agent = make_agent(agent_name, params)
    environment = TrafficEnvironment(
        config=rl_config,
        scenario=scenario,
        seed=seed,
        gui=gui,
        duration=duration,
    )

    run_dir = Path(output_dir) / agent_name
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = run_dir / f"train_{scenario}_seed{seed}_{stamp}.csv"
    checkpoint_dir = Path(save_dir) / agent_name / f"{scenario}_seed{seed}"

    started = time.perf_counter()
    last_episode: dict[str, Any] = {}
    try:
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
            writer.writeheader()
            for episode in range(n_episodes):
                episode_started = time.perf_counter()
                # Training seeds are offset away from the evaluation grid
                # (42-45) so no training episode is ever an evaluation run.
                # Demand arrivals are deterministic per scenario; the seed
                # varies driver behaviour only.
                episode_seed = seed + 1000 + episode
                observation = environment.reset(scenario=scenario, seed=episode_seed)
                agent.reset()
                total_reward = 0.0
                steps = 0
                while True:
                    action = agent.act(observation)
                    result = environment.step(action)
                    total_reward += result.reward
                    steps += 1
                    observation = result.observation
                    if result.terminated or result.truncated:
                        break
                metrics = dict(result.info.get("episode_metrics") or {})
                row = {
                    "episode": episode,
                    "seed": episode_seed,
                    "scenario": scenario,
                    "total_reward": round(total_reward, 4),
                    "steps": steps,
                    "mean_queue_total": metrics.get("mean_queue_total", ""),
                    "mean_waiting_total": metrics.get("mean_waiting_total", ""),
                    "throughput_veh_per_h": metrics.get("throughput_veh_per_h", ""),
                    "arrived_vehicles": metrics.get("arrived_vehicles", ""),
                    "mean_travel_time_s": metrics.get("mean_travel_time_s", ""),
                    "epsilon": round(float(getattr(agent, "epsilon", 0.0)), 6),
                    "wall_seconds": round(time.perf_counter() - episode_started, 3),
                }
                writer.writerow(row)
                handle.flush()
                last_episode = row
                if interval and (episode + 1) % interval == 0:
                    agent.save(checkpoint_dir)
                if (episode + 1) % max(1, n_episodes // 10) == 0 or episode == 0:
                    print(
                        f"[{agent_name}] episode {episode + 1}/{n_episodes}  "
                        f"reward {row['total_reward']}  "
                        f"queue {row['mean_queue_total']}  "
                        f"eps {row['epsilon']}"
                    )
        checkpoint_path = agent.save(checkpoint_dir)
    finally:
        environment.close()

    summary = {
        "agent": agent_name,
        "scenario": scenario,
        "seed": seed,
        "episodes": n_episodes,
        "state": params.get("state"),
        "duration": duration,
        "params": {key: value for key, value in params.items() if key != "seed"},
        "log": str(log_path),
        "checkpoint": str(checkpoint_path),
        "last_episode": last_episode,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    summary_path = run_dir / f"train_{scenario}_seed{seed}_{stamp}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    summary["summary"] = str(summary_path)
    print(
        f"Saved {n_episodes} episodes to {log_path} and checkpoint to {checkpoint_path}"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.agent not in available_agents():
        raise SystemExit(
            f"Unknown agent {args.agent!r}. Registered agents: "
            f"{', '.join(available_agents()) or '(none yet)'}. "
            "Register yours in src/reinforcement/agents.py."
        )
    rl_config = load_yaml(args.config)
    agents_config = load_yaml(args.agents_config) if args.agents_config.is_file() else {}
    train(
        agent_name=args.agent,
        rl_config=rl_config,
        agents_config=agents_config,
        scenario=args.scenario,
        seed=args.seed,
        episodes=args.episodes,
        duration=args.duration,
        state=args.state,
        save_every=args.save_every,
        output_dir=args.output_dir,
        save_dir=args.save_dir,
        gui=args.gui,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
