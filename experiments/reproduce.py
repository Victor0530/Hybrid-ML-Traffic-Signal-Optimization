"""Reproduce the complete M1-M3 pipeline from a clean checkout.

The script is intentionally a thin orchestrator: it calls the same modules the
team uses manually, skips valid existing runs, and stops on the first failure.
It does not contain a second implementation of route generation, simulation,
dataset construction or tests.

Typical use from the project root with the virtual environment active::

    python -m experiments.reproduce

    # Data already exists; rebuild only the baseline, dataset, EDA and tests
    python -m experiments.reproduce --skip-routes --skip-collection

    # Check the environment without running anything expensive
    python -m experiments.reproduce --check-only
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (42, 43, 44, 45)
REQUIRED_PACKAGES = ("traci", "sumolib", "pandas", "numpy", "sklearn", "yaml", "matplotlib")
NETWORK_CONFIG = PROJECT_ROOT / "sumo" / "network" / "intersection.sumocfg"
NETWORK_FILE = PROJECT_ROOT / "sumo" / "network" / "intersection.net.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="scenario names to generate/collect (default: every configured scenario)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="random seeds (default: 42 43 44 45)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-run simulations even when a valid raw CSV already exists",
    )
    parser.add_argument("--skip-routes", action="store_true", help="do not regenerate route files")
    parser.add_argument("--skip-collection", action="store_true", help="use existing raw CSVs")
    parser.add_argument("--skip-baseline", action="store_true", help="do not rerun M2 baseline analysis")
    parser.add_argument("--skip-dataset", action="store_true", help="do not rebuild the M3 dataset")
    parser.add_argument("--skip-eda", action="store_true", help="do not rerun the M3 EDA")
    parser.add_argument("--skip-tests", action="store_true", help="do not run pytest")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="check Python, packages, SUMO and the network, then exit",
    )
    return parser.parse_args()


def check_python() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            f"Python 3.10+ is required; this interpreter is {sys.version.split()[0]}."
        )
    print(f"[ok] Python {sys.version.split()[0]}")


def check_packages() -> None:
    missing = []
    for name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            missing.append(name)
            continue
        version = getattr(module, "__version__", "?")
        print(f"[ok] {name} {version}")
    if missing:
        raise RuntimeError(
            "Missing packages: "
            + ", ".join(missing)
            + ". Activate the virtual environment and run `pip install -r requirements.txt`."
        )


def check_sumo(required: bool) -> str | None:
    from src.simulation.sumo_binary import get_sumo_version, resolve_sumo_binary

    try:
        binary = resolve_sumo_binary("sumo")
        version = get_sumo_version(binary)
    except Exception as exc:  # noqa: BLE001 - report the environment failure clearly
        if required:
            raise RuntimeError(
                "SUMO could not be found. Install Eclipse SUMO, set SUMO_HOME, "
                "and make sure `sumo` is on PATH."
            ) from exc
        print(f"[warn] SUMO not available (collection is skipped): {exc}")
        return None
    print(f"[ok] SUMO {version} ({binary})")
    for path in (NETWORK_CONFIG, NETWORK_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing SUMO network file: {path}")
    print(f"[ok] network {NETWORK_CONFIG.relative_to(PROJECT_ROOT)}")
    return binary


def run_module(module: str, description: str, *arguments: str) -> None:
    print(f"\n=== {description} ===")
    subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    print("=== Environment checks ===")
    check_python()
    check_packages()

    from src.simulation.scenario_generator import load_config

    config = load_config()
    configured_scenarios = list(config["scenarios"])
    if args.scenarios:
        unknown = sorted(set(args.scenarios).difference(configured_scenarios))
        if unknown:
            raise ValueError(
                f"Unknown scenario(s): {', '.join(unknown)}. "
                f"Choose from: {', '.join(configured_scenarios)}"
            )
        scenarios = args.scenarios
    else:
        scenarios = configured_scenarios

    if not args.seeds:
        raise ValueError("At least one seed is required.")
    check_sumo(required=not args.skip_collection)
    if args.check_only:
        print("\nEnvironment check passed.")
        return 0

    if not args.skip_routes:
        from src.simulation.scenario_generator import write_scenario_routes

        print("\n=== Generate route files ===")
        for scenario in scenarios:
            path = write_scenario_routes(scenario, config)
            print(f"[ok] {path.relative_to(PROJECT_ROOT)}")

    if not args.skip_collection:
        from src.simulation.data_collector import collect_run

        duration = int(config["simulation"]["duration"])
        print(
            f"\n=== Collect SUMO runs ({len(scenarios)} scenarios x {len(args.seeds)} seeds) ==="
        )
        failed: list[str] = []
        for scenario in scenarios:
            for seed in args.seeds:
                run_id = f"{scenario}_seed{seed}"
                ok = collect_run(
                    scenario,
                    seed,
                    config,
                    duration,
                    overwrite=args.overwrite,
                )
                if not ok:
                    failed.append(run_id)
        if failed:
            raise RuntimeError(
                "Some runs failed or were invalid: "
                + ", ".join(failed)
                + ". Check the messages above; use --overwrite only after fixing the "
                "cause (for example a changed route or network hash)."
            )

    if not args.skip_baseline:
        run_module(
            "src.analysis.characterise_baseline",
            "M2 fixed-time baseline analysis",
        )
    if not args.skip_dataset:
        run_module("src.preprocessing.build_dataset", "M3 dataset build")
    if not args.skip_eda:
        run_module("src.analysis.explore_sl_dataset", "M3 exploratory data analysis")
    if not args.skip_tests:
        run_module("pytest", "Test suite", "-q")

    elapsed = time.perf_counter() - started
    print(f"\nDone in {elapsed:.1f}s.")
    print(
        "Outputs: data/raw/, results/fixed_time_baseline/, data/processed/, "
        "results/sl/eda/."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
