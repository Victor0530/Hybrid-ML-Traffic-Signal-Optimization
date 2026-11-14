"""Generate SUMO route files from the traffic scenario configuration.

Demand is defined per approach in vehicles per hour and split into straight,
left and right movements. Every scenario gets one deterministic route file.

Usage:
    python -m src.simulation.scenario_generator --scenario medium
    python -m src.simulation.scenario_generator --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "traffic_scenarios.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "sumo" / "routes" / "generated"

DIRECTIONS = ("north", "south", "east", "west")
MOVEMENTS = ("straight", "left", "right")

# For right-hand traffic: destination edge for each movement.
DESTINATIONS = {
    "north": {"straight": "south_out", "left": "east_out", "right": "west_out"},
    "south": {"straight": "north_out", "left": "west_out", "right": "east_out"},
    "east": {"straight": "west_out", "left": "south_out", "right": "north_out"},
    "west": {"straight": "east_out", "left": "north_out", "right": "south_out"},
}


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_demand_phases(scenario: dict, duration: float) -> list[dict]:
    """Return demand phases with explicit begin/end/demand keys."""
    if "phases" in scenario:
        return scenario["phases"]
    return [{"begin": 0, "end": duration, "demand": scenario["demand"]}]


def build_route_xml(scenario_name: str, config: dict) -> str:
    scenario = config["scenarios"][scenario_name]
    simulation = config["simulation"]
    duration = simulation["duration"]
    split = simulation["turning_split"]

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<!-- Generated from configs/traffic_scenarios.yaml: {scenario_name} -->",
        "<routes>",
        '    <vType id="car" vClass="passenger" accel="2.6" decel="4.5" sigma="0.5"',
        '           length="5.0" minGap="2.5" maxSpeed="13.89"/>',
        "",
    ]

    for direction in DIRECTIONS:
        for movement in MOVEMENTS:
            destination = DESTINATIONS[direction][movement]
            lines.append(
                f'    <route id="route_{direction}_{movement}" '
                f'edges="{direction}_in {destination}"/>'
            )
    lines.append("")

    for phase_index, phase in enumerate(get_demand_phases(scenario, duration)):
        begin, end = phase["begin"], phase["end"]
        lines.append(f"    <!-- Demand phase {phase_index}: {begin}-{end} s -->")
        for direction in DIRECTIONS:
            approach_rate = phase["demand"][direction]
            for movement in MOVEMENTS:
                rate = approach_rate * split[movement]
                if rate <= 0:
                    continue
                lines.append(
                    f'    <flow id="flow_{scenario_name}_{direction}_{movement}_{phase_index}" '
                    f'route="route_{direction}_{movement}" begin="{begin}" end="{end}" '
                    f'vehsPerHour="{rate:.1f}" departLane="best" departSpeed="max"/>'
                )
        lines.append("")

    lines.append("</routes>")
    return "\n".join(lines) + "\n"


def write_scenario_routes(
    scenario_name: str,
    config: dict | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    config = config or load_config()
    if scenario_name not in config["scenarios"]:
        raise KeyError(f"Unknown scenario: {scenario_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{scenario_name}.rou.xml"
    output_path.write_text(build_route_xml(scenario_name, config), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", choices=None, help="scenario name")
    group.add_argument("--all", action="store_true", help="generate every scenario")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    config = load_config(args.config)
    names = list(config["scenarios"]) if args.all else [args.scenario]
    if args.scenario and args.scenario not in config["scenarios"]:
        parser.error(
            f"unknown scenario {args.scenario!r}; "
            f"choose from {', '.join(config['scenarios'])}"
        )

    for name in names:
        path = write_scenario_routes(name, config, args.output_dir)
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
