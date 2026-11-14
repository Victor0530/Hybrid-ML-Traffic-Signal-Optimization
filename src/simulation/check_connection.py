"""Verify Python can control the SUMO intersection through TraCI.

Runs the medium-demand scenario for a few simulated minutes and prints the
signal phase and incoming vehicle counts once per minute.

Usage:
    python src/simulation/check_connection.py
    python src/simulation/check_connection.py --gui
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import traci
except ImportError:
    sys.exit(
        "TraCI is not importable. Install SUMO, set SUMO_HOME, and install "
        "the project requirements."
    )

from src.simulation.sumo_binary import resolve_sumo_binary

SUMO_CFG = PROJECT_ROOT / "sumo" / "network" / "intersection.sumocfg"
INCOMING_EDGES = ("north_in", "south_in", "east_in", "west_in")
TLS_ID = "C"

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="run sumo-gui")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--end",
        type=int,
        default=300,
        help="simulated seconds to run (default: 300)",
    )
    args = parser.parse_args()

    binary = resolve_sumo_binary("sumo-gui" if args.gui else "sumo")
    print(f"Starting {binary} with {SUMO_CFG.name}, seed={args.seed}")

    traci.start(
        [
            binary,
            "-c",
            str(SUMO_CFG),
            "--seed",
            str(args.seed),
            "--end",
            str(args.end),
            "--no-step-log",
            "true",
            "--duration-log.statistics",
            "true",
        ]
    )

    try:
        print(f"{'time':>6}  {'phase':>5}  {'state':>12}  " + "  ".join(f"{e:>10}" for e in INCOMING_EDGES))
        while traci.simulation.getTime() < args.end:
            traci.simulationStep()
            now = int(traci.simulation.getTime())
            if now % 60 != 0:
                continue
            phase = traci.trafficlight.getPhase(TLS_ID)
            state = traci.trafficlight.getRedYellowGreenState(TLS_ID)
            counts = [
                traci.edge.getLastStepVehicleNumber(edge) for edge in INCOMING_EDGES
            ]
            print(
                f"{now:>6}  {phase:>5}  {state:>12}  "
                + "  ".join(f"{count:>10}" for count in counts)
            )

        print(
            f"Finished at t={traci.simulation.getTime():.0f}s: "
            f"{traci.simulation.getArrivedNumber()} vehicles arrived in the final step, "
            f"{len(traci.vehicle.getIDList())} still running."
        )
    finally:
        traci.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
