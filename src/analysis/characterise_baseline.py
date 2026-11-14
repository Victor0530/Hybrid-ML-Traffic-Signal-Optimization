"""Characterise the fixed-time baseline from the observation dataset.

The script reads the run manifest and observation CSVs, builds 5-minute
time series per scenario and approach, computes throughput from the per-run
departure/arrival counters, and writes summary tables and plots for the
fixed-time environment analysis.

Usage:
    python -m src.analysis.characterise_baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.simulation.scenario_generator import DIRECTIONS, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "fixed_time_baseline"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "traffic_scenarios.yaml"
BIN_SECONDS = 300

COLORS = {
    "north": "#1f77b4",
    "south": "#ff7f0e",
    "east": "#2ca02c",
    "west": "#d62728",
}


def load_observations(data_dir: Path, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load canonical runs listed as successful in the manifest."""
    simulation = config["simulation"]
    duration = int(simulation["duration"])
    interval = int(simulation["sampling_interval"])

    manifest_path = data_dir / "run_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    required_manifest = {"run_id", "status", "duration", "sampling_interval"}
    missing = required_manifest - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    runs = manifest[
        (manifest["status"] == "ok")
        & (manifest["duration"] == duration)
        & (manifest["sampling_interval"] == interval)
    ].copy()
    if runs.empty:
        raise ValueError(
            "No successful runs match the configured duration and sampling interval."
        )

    frames = []
    for run_id in runs["run_id"]:
        path = data_dir / f"{run_id}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Manifest lists {run_id}, but {path} is missing.")
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    required = {
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
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "Observation CSVs are missing columns: "
            f"{sorted(missing)}. Rerun the collector with --overwrite."
        )

    numeric = [
        "simulation_time",
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
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data, runs


def add_bins(data: pd.DataFrame, bin_seconds: int) -> pd.DataFrame:
    """Assign samples to left-closed, right-open time bins."""
    data = data.copy()
    data["time_bin"] = ((data["simulation_time"] - 1) // bin_seconds) * bin_seconds
    return data


def build_time_series(data: pd.DataFrame, bin_seconds: int) -> pd.DataFrame:
    per_seed = (
        data.groupby(
            ["scenario", "seed", "direction", "time_bin"], as_index=False
        )
        .agg(
            mean_queue=("queue_length", "mean"),
            mean_waiting_time=("waiting_time", "mean"),
            mean_speed=("mean_speed", "mean"),
            mean_vehicle_count=("vehicle_count", "mean"),
            departed_count=("departed_count", "sum"),
            arrived_count=("arrived_count", "sum"),
            samples=("simulation_time", "count"),
        )
    )
    per_seed["departed_veh_per_h"] = (
        per_seed["departed_count"] * 3600.0 / bin_seconds
    )
    per_seed["arrived_veh_per_h"] = per_seed["arrived_count"] * 3600.0 / bin_seconds

    time_series = (
        per_seed.groupby(["scenario", "direction", "time_bin"], as_index=False)
        .agg(
            mean_queue=("mean_queue", "mean"),
            mean_waiting_time=("mean_waiting_time", "mean"),
            mean_speed=("mean_speed", "mean"),
            mean_vehicle_count=("mean_vehicle_count", "mean"),
            departed_veh_per_h=("departed_veh_per_h", "mean"),
            arrived_veh_per_h=("arrived_veh_per_h", "mean"),
            seeds=("seed", "nunique"),
        )
    )
    time_series["time_bin_start"] = time_series["time_bin"]
    time_series["time_bin_end"] = time_series["time_bin"] + bin_seconds
    return time_series


def demand_table(config: dict) -> pd.DataFrame:
    simulation = config["simulation"]
    duration = float(simulation["duration"])
    rows = []
    for scenario, spec in config["scenarios"].items():
        phases = spec.get("phases") or [
            {"begin": 0, "end": duration, "demand": spec["demand"]}
        ]
        total_duration = sum(
            float(phase["end"]) - float(phase["begin"]) for phase in phases
        )
        for direction in DIRECTIONS:
            mean_demand = (
                sum(
                    float(phase["demand"][direction])
                    * (float(phase["end"]) - float(phase["begin"]))
                    for phase in phases
                )
                / total_duration
            )
            peak_demand = max(
                float(phase["demand"][direction]) for phase in phases
            )
            rows.append(
                {
                    "scenario": scenario,
                    "direction": direction,
                    "demand_mean_veh_per_h": mean_demand,
                    "demand_peak_veh_per_h": peak_demand,
                }
            )
    return pd.DataFrame(rows)


def build_summary(
    data: pd.DataFrame,
    time_series: pd.DataFrame,
    config: dict,
    bin_seconds: int = BIN_SECONDS,
) -> pd.DataFrame:
    duration = float(config["simulation"]["duration"])
    last_window = duration - 600.0

    state = (
        data.groupby(["scenario", "direction"], as_index=False)
        .agg(
            queue_mean=("queue_length", "mean"),
            queue_max=("queue_length", "max"),
            waiting_time_mean=("waiting_time", "mean"),
            waiting_time_max=("waiting_time", "max"),
            mean_speed_mean=("mean_speed", "mean"),
            mean_speed_min=("mean_speed", "min"),
            vehicle_count_mean=("vehicle_count", "mean"),
            empty_share=("no_vehicles", "mean"),
            sensor_missing_share=("sensor_missing", "mean"),
        )
    )

    totals = (
        data.groupby(["scenario", "seed", "direction"], as_index=False)
        .agg(
            departed_total=("departed_count", "sum"),
            arrived_total=("arrived_count", "sum"),
        )
        .groupby(["scenario", "direction"], as_index=False)
        .agg(
            departed_total=("departed_total", "mean"),
            arrived_total=("arrived_total", "mean"),
        )
    )

    first = (
        data[data["simulation_time"] <= 600.0]
        .groupby(["scenario", "direction"], as_index=False)["queue_length"]
        .mean()
        .rename(columns={"queue_length": "queue_first10_mean"})
    )
    last = (
        data[data["simulation_time"] > last_window]
        .groupby(["scenario", "direction"], as_index=False)
        .agg(
            queue_last10_mean=("queue_length", "mean"),
            speed_last10_mean=("mean_speed", "mean"),
        )
    )

    peak = (
        time_series.sort_values("mean_queue")
        .groupby(["scenario", "direction"], as_index=False)
        .tail(1)[["scenario", "direction", "mean_queue", "time_bin"]]
        .rename(
            columns={
                "mean_queue": "peak_queue_5min",
                "time_bin": "peak_queue_bin_start",
            }
        )
    )

    summary = (
        state.merge(totals, on=["scenario", "direction"], how="left")
        .merge(first, on=["scenario", "direction"], how="left")
        .merge(last, on=["scenario", "direction"], how="left")
        .merge(peak, on=["scenario", "direction"], how="left")
        .merge(demand_table(config), on=["scenario", "direction"], how="left")
    )
    summary["throughput_veh_per_h"] = summary["arrived_total"] / duration * 3600.0
    summary["throughput_ratio"] = (
        summary["throughput_veh_per_h"] / summary["demand_mean_veh_per_h"]
    )
    summary["queue_growth"] = (
        summary["queue_last10_mean"] - summary["queue_first10_mean"]
    )
    summary["saturated_end"] = (
        (summary["queue_last10_mean"] >= 15.0)
        & (summary["speed_last10_mean"] <= 3.5)
    )
    summary["demand_underserved"] = summary["throughput_ratio"] < 0.90

    imbalance = (
        summary.groupby("scenario")["queue_last10_mean"]
        .agg(lambda values: values.max() - values.min())
        .rename("queue_imbalance_scenario")
    )
    summary = summary.merge(imbalance, on="scenario", how="left")

    changing_phases = config["scenarios"]["changing"]["phases"]
    peak_phase = max(
        changing_phases,
        key=lambda phase: max(phase["demand"].values()),
    )
    peak_end = float(peak_phase["end"])
    recovery_phase = next(
        (phase for phase in changing_phases if phase["begin"] >= peak_end),
        None,
    )
    if recovery_phase is None:
        summary["recovery_drop"] = float("nan")
    else:
        high_bin = ((peak_end - 1) // bin_seconds) * bin_seconds
        final_bin = (
            (float(recovery_phase["end"]) - 1) // bin_seconds
        ) * bin_seconds
        changing_high = time_series[
            (time_series["scenario"] == "changing")
            & (time_series["time_bin"] == high_bin)
        ].set_index(["scenario", "direction"])["mean_queue"]
        changing_final = time_series[
            (time_series["scenario"] == "changing")
            & (time_series["time_bin"] == final_bin)
        ].set_index(["scenario", "direction"])["mean_queue"]
        recovery = (
            (changing_high - changing_final)
            .rename("recovery_drop")
            .reset_index()
        )
        summary = summary.merge(
            recovery, on=["scenario", "direction"], how="left"
        )

    numeric = summary.select_dtypes(include="number").columns
    summary[numeric] = summary[numeric].round(3)
    return summary


def data_quality_table(
    data: pd.DataFrame, runs: pd.DataFrame
) -> pd.DataFrame:
    duplicate_keys = int(
        data.duplicated(["run_id", "simulation_time", "direction"]).sum()
    )
    arrived_unmapped = (
        int(runs["arrived_unmapped"].sum())
        if "arrived_unmapped" in runs.columns
        else 0
    )
    rows = [
        ("runs", len(runs)),
        ("observation_rows", len(data)),
        ("scenarios", data["scenario"].nunique()),
        ("seeds", data["seed"].nunique()),
        ("missing_values_total", int(data.isna().sum().sum())),
        ("mean_speed_missing", int(data["mean_speed"].isna().sum())),
        ("zero_vehicle_rows", int((data["vehicle_count"] == 0).sum())),
        ("no_vehicles_rows", int(data["no_vehicles"].sum())),
        ("sensor_missing_rows", int(data["sensor_missing"].sum())),
        ("duplicate_keys", duplicate_keys),
        ("arrived_unmapped", arrived_unmapped),
        ("mean_speed_min", round(float(data["mean_speed"].min()), 3)),
        ("mean_speed_max", round(float(data["mean_speed"].max()), 3)),
        ("occupancy_min", round(float(data["occupancy"].min()), 3)),
        ("occupancy_max", round(float(data["occupancy"].max()), 3)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def plot_metric(
    time_series: pd.DataFrame,
    scenarios: list[str],
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(scenarios), 1, figsize=(12, 2.4 * len(scenarios)), sharex=True
    )
    if len(scenarios) == 1:
        axes = [axes]

    for axis, scenario in zip(axes, scenarios):
        subset = time_series[time_series["scenario"] == scenario]
        for direction in DIRECTIONS:
            line = subset[subset["direction"] == direction]
            axis.plot(
                line["time_bin"] / 60.0,
                line[metric],
                label=direction,
                color=COLORS[direction],
                linewidth=1.5,
            )
        axis.set_title(scenario)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        if scenario == "changing":
            axis.axvline(30, color="black", linestyle="--", linewidth=1)
            axis.axvline(45, color="black", linestyle="--", linewidth=1)

    axes[-1].set_xlabel("Simulation time (min)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4)
    figure.suptitle(title, y=0.995)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_throughput_bars(
    summary: pd.DataFrame, scenarios: list[str], path: Path
) -> None:
    pivot = summary.pivot(
        index="scenario", columns="direction", values="throughput_veh_per_h"
    ).reindex(scenarios)
    axis = pivot.plot(kind="bar", figsize=(11, 5), color=[COLORS[d] for d in pivot.columns])
    axis.set_title("Average throughput by approach")
    axis.set_ylabel("Vehicles per hour")
    axis.set_xlabel("Scenario")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(title="Approach", ncol=4)
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=150)
    plt.close(axis.figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bin-seconds", type=int, default=BIN_SECONDS, help="time-series bin width"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    scenarios = list(config["scenarios"])
    data, runs = load_observations(args.data_dir, config)
    data = add_bins(data, args.bin_seconds)
    time_series = build_time_series(data, args.bin_seconds)
    summary = build_summary(data, time_series, config, args.bin_seconds)
    quality = data_quality_table(data, runs)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "baseline_summary.csv"
    timeseries_path = output_dir / "baseline_timeseries.csv"
    quality_path = output_dir / "data_quality_summary.csv"
    summary.to_csv(summary_path, index=False)
    time_series.to_csv(timeseries_path, index=False)
    quality.to_csv(quality_path, index=False)

    plot_metric(
        time_series,
        scenarios,
        "mean_queue",
        "Queue (vehicles/approach)",
        "Fixed-time baseline: queue length",
        output_dir / "queue_timeseries.png",
    )
    plot_metric(
        time_series,
        scenarios,
        "mean_waiting_time",
        "Waiting time (s)",
        "Fixed-time baseline: waiting time",
        output_dir / "waiting_timeseries.png",
    )
    plot_metric(
        time_series,
        scenarios,
        "mean_speed",
        "Mean speed (m/s)",
        "Fixed-time baseline: mean speed",
        output_dir / "speed_timeseries.png",
    )
    plot_metric(
        time_series,
        scenarios,
        "arrived_veh_per_h",
        "Arriving vehicles (veh/h)",
        "Fixed-time baseline: throughput by approach",
        output_dir / "throughput_timeseries.png",
    )
    plot_throughput_bars(
        summary, scenarios, output_dir / "throughput_by_approach.png"
    )

    print(f"Loaded {len(runs)} runs and {len(data):,} observation rows.")
    print("\nData quality summary:")
    print(quality.to_string(index=False))
    columns = [
        "scenario",
        "direction",
        "queue_mean",
        "queue_last10_mean",
        "waiting_time_mean",
        "mean_speed_mean",
        "throughput_veh_per_h",
        "saturated_end",
    ]
    print("\nBaseline summary:")
    print(summary[columns].to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {timeseries_path}")
    print(f"Wrote {quality_path}")
    print(f"Wrote plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())