"""Collect traffic observations from SUMO for supervised-learning experiments.

One CSV per run is written to data/raw/<run_id>.csv, with one row per
(sampling time, approach). A run manifest records the configuration and
status for reproducibility.

Usage:
    python -m src.simulation.data_collector --scenario medium --seed 42
    python -m src.simulation.data_collector --seed 42 --seed 43
    python -m src.simulation.data_collector --duration 300
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import traci

from src.simulation.scenario_generator import (
    DEFAULT_CONFIG,
    DIRECTIONS,
    load_config,
    write_scenario_routes,
)
from src.simulation.sumo_binary import get_sumo_version, resolve_sumo_binary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = DATA_RAW / "run_manifest.csv"
NETWORK_PATH = PROJECT_ROOT / "sumo" / "network" / "intersection.net.xml"
TLS_ID = "C"

OBSERVATION_FIELDS = [
    "run_id",
    "scenario",
    "seed",
    "simulation_time",
    "direction",
    "vehicle_count",
    "mean_speed",
    "queue_length",
    "waiting_time",
    "occupancy",
    "traffic_light_phase",
    "phase_duration",
    "time_in_phase",
    "no_vehicles",
    "sensor_missing",
    "departed_count",
    "arrived_count",
]

MANIFEST_FIELDS = [
    "run_id",
    "scenario",
    "seed",
    "status",
    "started_at",
    "finished_at",
    "simulation_end",
    "duration",
    "sampling_interval",
    "prediction_horizon",
    "route_file",
    "config_hash",
    "route_hash",
    "network_hash",
    "sumo_version",
    "loaded_total",
    "departed_total",
    "arrived_total",
    "running_total",
    "teleports",
    "arrived_unmapped",
    "error",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_network_hash(path: Path) -> str:
    """Hash the network's canonical XML, ignoring netconvert's timestamp comment.

    netconvert writes a generation timestamp into the XML header, so hashing the
    raw file would make an otherwise identical network look changed after every
    rebuild.
    """
    root = ET.parse(path).getroot()
    canonical = ET.tostring(root, encoding="utf-8")
    return hashlib.sha256(canonical).hexdigest()


def hash_config(config: dict) -> str:
    """Hash the effective configuration, not the YAML formatting."""
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_manifest() -> dict[str, dict]:
    if not MANIFEST_PATH.exists():
        return {}
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as handle:
        return {row["run_id"]: row for row in csv.DictReader(handle)}


def upsert_manifest(row: dict) -> None:
    """Insert or replace one run in the manifest CSV atomically."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as handle:
            rows = [
                existing
                for existing in csv.DictReader(handle)
                if existing.get("run_id") != row["run_id"]
            ]
    rows.append(row)
    rows.sort(key=lambda item: item["run_id"])

    temp_path = MANIFEST_PATH.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, MANIFEST_PATH)


def validate_run_csv(
    path: Path, run_id: str, duration: int, interval: int
) -> tuple[bool, str]:
    """Check that an existing run CSV is complete and has the current schema."""
    if not path.exists():
        return False, "missing"
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != OBSERVATION_FIELDS:
                return False, "column schema changed"
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        return False, f"could not read: {exc}"

    expected_times = set(range(interval, duration + 1, interval))
    expected_rows = len(expected_times) * len(DIRECTIONS)
    if len(rows) != expected_rows:
        return False, f"row count {len(rows)} != {expected_rows}"

    times = set()
    directions = set()
    for row in rows:
        if row["run_id"] != run_id:
            return False, "run_id mismatch"
        try:
            times.add(int(row["simulation_time"]))
        except (TypeError, ValueError):
            return False, "invalid simulation_time"
        directions.add(row["direction"])

    if times != expected_times:
        return False, "timestamp coverage mismatch"
    if directions != set(DIRECTIONS):
        return False, "direction mismatch"
    return True, "ok"


def direction_for_vehicle(vehicle_id: str) -> str | None:
    """Map a departing vehicle to its approach from its current edge or route."""
    road = traci.vehicle.getRoadID(vehicle_id)
    for direction in DIRECTIONS:
        if road == f"{direction}_in":
            return direction

    route = traci.vehicle.getRoute(vehicle_id)
    if route:
        first_edge = route[0]
        if first_edge.endswith("_in"):
            return first_edge[: -len("_in")]
    return None


def sample_observations(
    run_id: str,
    scenario: str,
    seed: int,
    simulation_time: int,
    departed_counts: dict[str, int],
    arrived_counts: dict[str, int],
) -> list[dict]:
    """Build one row per approach for the current simulation step."""
    phase = traci.trafficlight.getPhase(TLS_ID)
    phase_duration = traci.trafficlight.getPhaseDuration(TLS_ID)
    next_switch = traci.trafficlight.getNextSwitch(TLS_ID)
    time_in_phase = max(0.0, phase_duration - max(0.0, next_switch - simulation_time))

    rows = []
    for direction in DIRECTIONS:
        edge = f"{direction}_in"
        vehicle_count = traci.edge.getLastStepVehicleNumber(edge)
        mean_speed = traci.edge.getLastStepMeanSpeed(edge)

        if vehicle_count == 0:
            no_vehicles, sensor_missing = 1, 0
            mean_speed_value = ""
        elif mean_speed < 0 or math.isnan(mean_speed):
            no_vehicles, sensor_missing = 0, 1
            mean_speed_value = ""
        else:
            no_vehicles, sensor_missing = 0, 0
            mean_speed_value = f"{mean_speed:.3f}"

        rows.append(
            {
                "run_id": run_id,
                "scenario": scenario,
                "seed": seed,
                "simulation_time": simulation_time,
                "direction": direction,
                "vehicle_count": vehicle_count,
                "mean_speed": mean_speed_value,
                "queue_length": traci.edge.getLastStepHaltingNumber(edge),
                "waiting_time": f"{traci.edge.getWaitingTime(edge):.3f}",
                "occupancy": f"{traci.edge.getLastStepOccupancy(edge):.3f}",
                "traffic_light_phase": phase,
                "phase_duration": f"{phase_duration:.1f}",
                "time_in_phase": f"{time_in_phase:.1f}",
                "no_vehicles": no_vehicles,
                "sensor_missing": sensor_missing,
                "departed_count": departed_counts.get(direction, 0),
                "arrived_count": arrived_counts.get(direction, 0),
            }
        )
    return rows


def collect_run(
    scenario: str,
    seed: int,
    config: dict,
    duration: int,
    gui: bool = False,
    overwrite: bool = False,
) -> bool:
    simulation = config["simulation"]
    interval = int(simulation["sampling_interval"])
    horizon = int(simulation["prediction_horizon"])
    default_duration = int(simulation["duration"])

    run_id = f"{scenario}_seed{seed}"
    if duration != default_duration:
        run_id = f"{run_id}_d{duration}"

    output_path = DATA_RAW / f"{run_id}.csv"
    route_path = (
        PROJECT_ROOT / "sumo" / "routes" / "generated" / f"{scenario}.rou.xml"
    )
    config_hash = hash_config(config)

    if output_path.exists() and not overwrite:
        valid, reason = validate_run_csv(output_path, run_id, duration, interval)
        manifest_row = read_manifest().get(run_id)
        if not valid:
            print(
                f"[invalid] {run_id}: {reason}; rerun with --overwrite",
                file=sys.stderr,
            )
            return False
        if manifest_row is None:
            print(
                f"[invalid] {run_id}: missing manifest row; rerun with --overwrite",
                file=sys.stderr,
            )
            return False
        if manifest_row.get("status") != "ok":
            print(
                f"[invalid] {run_id}: manifest status is "
                f"{manifest_row.get('status')!r}; rerun with --overwrite",
                file=sys.stderr,
            )
            return False
        current_route_hash = sha256_file(route_path) if route_path.exists() else ""
        current_network_hash = (
            stable_network_hash(NETWORK_PATH) if NETWORK_PATH.exists() else ""
        )
        for label, stored, current in (
            ("configuration", manifest_row.get("config_hash"), config_hash),
            ("route", manifest_row.get("route_hash"), current_route_hash),
            ("network", manifest_row.get("network_hash"), current_network_hash),
        ):
            if stored in (None, ""):
                print(
                    f"[invalid] {run_id}: manifest {label} hash is missing; "
                    "rerun with --overwrite",
                    file=sys.stderr,
                )
                return False
            if stored != current:
                print(
                    f"[invalid] {run_id}: {label} changed; "
                    "rerun with --overwrite",
                    file=sys.stderr,
                )
                return False
        print(f"[skip] {run_id}: {output_path} already exists (use --overwrite)")
        return True

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".csv.tmp")
    if temp_path.exists():
        temp_path.unlink()

    started_at = utc_now()
    status = "failed"
    error = ""
    simulation_end = 0.0
    loaded_total = 0
    departed_total = 0
    arrived_total = 0
    running_total = 0
    teleports = 0
    arrived_unmapped = 0
    route_hash = ""
    network_hash = ""
    sumo_version = ""

    print(f"[run]  {run_id}: {duration}s, sampling every {interval}s")

    try:
        route_path = write_scenario_routes(scenario, config)
        route_hash = sha256_file(route_path)
        network_hash = stable_network_hash(NETWORK_PATH)
        binary = resolve_sumo_binary("sumo-gui" if gui else "sumo")
        sumo_version = get_sumo_version(binary)

        traci.start(
            [
                binary,
                "-c",
                str(PROJECT_ROOT / "sumo" / "network" / "intersection.sumocfg"),
                "--route-files",
                str(route_path),
                "--seed",
                str(seed),
                "--end",
                str(duration),
                "--no-step-log",
                "true",
            ]
        )

        vehicle_approach: dict[str, str] = {}
        departed_interval = {direction: 0 for direction in DIRECTIONS}
        arrived_interval = {direction: 0 for direction in DIRECTIONS}

        with temp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OBSERVATION_FIELDS)
            writer.writeheader()

            while traci.simulation.getTime() < duration:
                traci.simulationStep()
                simulation_end = traci.simulation.getTime()
                loaded_total += traci.simulation.getLoadedNumber()
                teleports += traci.simulation.getStartingTeleportNumber()

                for vehicle_id in traci.simulation.getDepartedIDList():
                    direction = direction_for_vehicle(vehicle_id)
                    departed_total += 1
                    if direction is not None:
                        vehicle_approach[vehicle_id] = direction
                        departed_interval[direction] += 1

                for vehicle_id in traci.simulation.getArrivedIDList():
                    arrived_total += 1
                    direction = vehicle_approach.pop(vehicle_id, None)
                    if direction is None:
                        arrived_unmapped += 1
                    else:
                        arrived_interval[direction] += 1

                for vehicle_id in traci.simulation.getStartingTeleportIDList():
                    vehicle_approach.pop(vehicle_id, None)

                step = int(simulation_end)
                if step % interval == 0:
                    writer.writerows(
                        sample_observations(
                            run_id,
                            scenario,
                            seed,
                            step,
                            departed_interval,
                            arrived_interval,
                        )
                    )
                    departed_interval = {direction: 0 for direction in DIRECTIONS}
                    arrived_interval = {direction: 0 for direction in DIRECTIONS}

        running_total = len(traci.vehicle.getIDList())
        os.replace(temp_path, output_path)
        status = "ok"
        print(f"[done] {run_id}: {output_path}")
    except KeyboardInterrupt:
        error = "KeyboardInterrupt"
        print(f"\n[interrupt] {run_id}: stopped by user", file=sys.stderr)
        raise
    except Exception as exc:  # noqa: BLE001 - record any run failure in the manifest
        error = f"{type(exc).__name__}: {exc}"
        print(f"[fail] {run_id}: {error}", file=sys.stderr)
        traceback.print_exc()
    finally:
        try:
            traci.close()
        except Exception:  # noqa: BLE001 - closing must not hide the real error
            pass
        if temp_path.exists():
            temp_path.unlink()
        try:
            upsert_manifest(
                {
                    "run_id": run_id,
                    "scenario": scenario,
                    "seed": seed,
                    "status": status,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "simulation_end": f"{simulation_end:.1f}",
                    "duration": duration,
                    "sampling_interval": interval,
                    "prediction_horizon": horizon,
                    "route_file": route_path.relative_to(PROJECT_ROOT).as_posix(),
                    "config_hash": config_hash,
                    "route_hash": route_hash,
                    "network_hash": network_hash,
                    "sumo_version": sumo_version,
                    "loaded_total": loaded_total,
                    "departed_total": departed_total,
                    "arrived_total": arrived_total,
                    "running_total": running_total,
                    "teleports": teleports,
                    "arrived_unmapped": arrived_unmapped,
                    "error": error,
                }
            )
        except Exception as exc:  # noqa: BLE001 - never hide the simulation result
            print(f"[warn] could not update manifest: {exc}", file=sys.stderr)

    return status == "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        action="append",
        help="scenario to run (repeatable; default: all scenarios)",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="random seed (repeatable; default: 42)",
    )
    parser.add_argument("--duration", type=int, help="override simulation duration")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gui", action="store_true", help="show sumo-gui")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing run CSV files"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    scenarios = args.scenario or list(config["scenarios"])
    unknown = [name for name in scenarios if name not in config["scenarios"]]
    if unknown:
        parser.error(
            f"unknown scenario(s) {unknown}; choose from {', '.join(config['scenarios'])}"
        )
    seeds = args.seed or [42]
    duration = args.duration or int(config["simulation"]["duration"])

    try:
        results = [
            collect_run(
                scenario,
                seed,
                config,
                duration,
                gui=args.gui,
                overwrite=args.overwrite,
            )
            for scenario in scenarios
            for seed in seeds
        ]
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} runs completed successfully")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())