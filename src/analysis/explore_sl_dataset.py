"""Exploratory data analysis for the M3 supervised-learning dataset.

The script reads the processed long-format dataset and writes report-ready
tables and figures to ``results/sl/eda/``. It never modifies the dataset.

Run from the project root::

    python -m src.analysis.explore_sl_dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "sl_dataset.csv"
DEFAULT_FEATURE_MAP = PROJECT_ROOT / "data" / "processed" / "feature_columns.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "sl" / "eda"

TARGET = "target_vehicle_count"
DIRECTIONS = ("north", "south", "east", "west")
SPLITS = ("train", "validation", "test")
FEATURE_SETS = ("F0", "F1", "F2", "F3", "F4")
AUTOCORRELATION_LAGS = (1, 3, 6, 12, 36, 60)


def load_dataset(path: Path = DEFAULT_DATASET) -> pd.DataFrame:
    """Load the processed dataset and validate the columns needed for EDA."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run `python -m src.preprocessing.build_dataset` first."
        )
    data = pd.read_csv(path)
    required = {
        "run_id",
        "scenario",
        "seed",
        "simulation_time",
        "direction",
        "split",
        "vehicle_count",
        TARGET,
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("Dataset is empty.")
    return data


def load_feature_map(path: Path = DEFAULT_FEATURE_MAP) -> dict[str, list[str]]:
    """Load the ordered F0-F4 feature map written by the dataset builder."""
    if not path.exists():
        raise FileNotFoundError(
            f"Feature map not found: {path}. Run `python -m src.preprocessing.build_dataset` first."
        )
    feature_map = json.loads(path.read_text(encoding="utf-8"))
    missing = [name for name in FEATURE_SETS if name not in feature_map]
    if missing:
        raise ValueError(f"Feature map is missing sets: {', '.join(missing)}")
    return {name: list(feature_map[name]) for name in FEATURE_SETS}


def feature_set_lookup(feature_map: dict[str, list[str]]) -> dict[str, str]:
    """Map each feature to the first (smallest) feature set that contains it."""
    lookup: dict[str, str] = {}
    for name in FEATURE_SETS:
        for feature in feature_map[name]:
            lookup.setdefault(feature, name)
    return lookup


def safe_correlation(left: pd.Series, right: pd.Series) -> float:
    """Pearson correlation that returns NaN instead of warning for constants."""
    left = pd.to_numeric(left, errors="coerce")
    right = pd.to_numeric(right, errors="coerce")
    valid = left.notna() & right.notna()
    if int(valid.sum()) < 2:
        return float("nan")
    left_valid = left[valid]
    right_valid = right[valid]
    if left_valid.std(ddof=0) == 0 or right_valid.std(ddof=0) == 0:
        return float("nan")
    return float(left_valid.corr(right_valid))


def distribution_summary(
    data: pd.DataFrame,
    group_columns: list[str],
    value: str = TARGET,
) -> pd.DataFrame:
    """Summarise a numeric column by one or more grouping columns."""
    grouped = data.groupby(group_columns, observed=True)[value]
    summary = grouped.agg(
        rows="count",
        mean="mean",
        std="std",
        minimum="min",
        q25=lambda values: values.quantile(0.25),
        median="median",
        q75=lambda values: values.quantile(0.75),
        maximum="max",
    ).reset_index()
    return summary


def feature_summary(
    data: pd.DataFrame,
    features: list[str],
    lookup: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Describe every feature, including its earliest feature set and missingness."""
    lookup = lookup or {}
    rows: list[dict[str, object]] = []
    for feature in features:
        values = pd.to_numeric(data[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "feature_set": lookup.get(feature, ""),
                "rows": int(values.notna().sum()),
                "missing_pct": float(values.isna().mean() * 100.0),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "minimum": float(values.min()),
                "q25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "q75": float(values.quantile(0.75)),
                "maximum": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def correlation_table(
    data: pd.DataFrame,
    features: list[str],
    lookup: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Pearson correlation of every feature with the target, overall and by split."""
    lookup = lookup or {}
    rows: list[dict[str, object]] = []
    for feature in features:
        row: dict[str, object] = {
            "feature": feature,
            "feature_set": lookup.get(feature, ""),
            "corr_all": safe_correlation(data[feature], data[TARGET]),
        }
        for split in SPLITS:
            subset = data[data["split"] == split]
            row[f"corr_{split}"] = safe_correlation(subset[feature], subset[TARGET])
        rows.append(row)
    table = pd.DataFrame(rows)
    return table.reindex(
        table["corr_all"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def autocorrelation_table(
    data: pd.DataFrame,
    lags: tuple[int, ...] = AUTOCORRELATION_LAGS,
) -> pd.DataFrame:
    """Autocorrelation of vehicle_count within each run and direction.

    Lags are shifted inside each ``(run_id, direction)`` group so pairs never
    cross a run boundary. Correlations are computed on the pooled valid pairs.
    """
    ordered = data.sort_values(["run_id", "direction", "simulation_time"])
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("all", ordered)]
    scopes.extend((direction, ordered[ordered["direction"] == direction]) for direction in DIRECTIONS)

    for scope, subset in scopes:
        for lag in lags:
            shifted = subset.groupby(["run_id", "direction"], observed=True)[
                "vehicle_count"
            ].shift(lag)
            valid = shifted.notna()
            correlation = safe_correlation(
                subset.loc[valid, "vehicle_count"], shifted[valid]
            )
            rows.append(
                {
                    "scope": scope,
                    "lag_samples": lag,
                    "lag_seconds": lag * 5,
                    "pairs": int(valid.sum()),
                    "autocorrelation": correlation,
                }
            )
    return pd.DataFrame(rows)


def split_shift_table(
    data: pd.DataFrame,
    features: list[str],
    lookup: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Standardised mean differences between train and validation/test.

    This is a descriptive effect size, not a hypothesis test. Values above
    roughly 0.2 indicate a non-trivial distribution shift.
    """
    lookup = lookup or {}
    reference = data[data["split"] == "train"]
    rows: list[dict[str, object]] = []
    for column in features + [TARGET]:
        row: dict[str, object] = {
            "column": column,
            "feature_set": lookup.get(column, "target" if column == TARGET else ""),
            "mean_train": reference[column].mean(),
        }
        variances = [reference[column].var()]
        for split in ("validation", "test"):
            subset = data[data["split"] == split]
            row[f"mean_{split}"] = subset[column].mean()
            variances.append(subset[column].var())
        pooled = float(np.sqrt(np.nanmean(variances)))
        for split in ("validation", "test"):
            difference = row[f"mean_{split}"] - row["mean_train"]
            row[f"smd_train_{split}"] = (
                float(difference / pooled) if pooled > 0 else np.nan
            )
        row["max_abs_smd"] = max(
            (
                abs(value)
                for value in (row["smd_train_validation"], row["smd_train_test"])
                if pd.notna(value)
            ),
            default=float("nan"),
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    return table.sort_values("max_abs_smd", ascending=False).reset_index(drop=True)


def phase_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Target distribution by active traffic-light phase."""
    summary = distribution_summary(data, ["traffic_light_phase"])
    counts = data["traffic_light_phase"].value_counts().rename("share_pct")
    summary["share_pct"] = (
        summary["traffic_light_phase"].map(counts) / len(data) * 100.0
    )
    return summary


def build_overview(
    data: pd.DataFrame,
    feature_map: dict[str, list[str]],
) -> dict[str, object]:
    """Return the high-level facts used in the EDA report."""
    return {
        "rows": int(len(data)),
        "runs": int(data["run_id"].nunique()),
        "scenarios": sorted(data["scenario"].unique().tolist()),
        "seeds": sorted(int(seed) for seed in data["seed"].unique()),
        "directions": sorted(data["direction"].unique().tolist()),
        "splits": {split: int((data["split"] == split).sum()) for split in SPLITS},
        "simulation_time": {
            "minimum": int(data["simulation_time"].min()),
            "maximum": int(data["simulation_time"].max()),
            "step_seconds": int(
                data.sort_values(["run_id", "direction", "simulation_time"])
                .groupby(["run_id", "direction"], observed=True)["simulation_time"]
                .diff()
                .dropna()
                .min()
            ),
        },
        "feature_counts": {name: len(columns) for name, columns in feature_map.items()},
        "missing_cells": int(data.isna().sum().sum()),
        "duplicate_keys": int(
            data.duplicated(["run_id", "simulation_time", "direction"]).sum()
        ),
        "zero_vehicle_pct": float((data["vehicle_count"] == 0).mean() * 100.0),
        "target": {
            "mean": float(data[TARGET].mean()),
            "std": float(data[TARGET].std()),
            "minimum": float(data[TARGET].min()),
            "maximum": float(data[TARGET].max()),
        },
    }


def make_plots(
    data: pd.DataFrame,
    correlations: pd.DataFrame,
    shifts: pd.DataFrame,
    output_dir: Path,
    run_id: str | None = None,
) -> list[Path]:
    """Write the EDA figures and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(figure: plt.Figure, name: str) -> None:
        path = output_dir / name
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path)

    # Target distribution by split.
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for split in SPLITS:
        values = data.loc[data["split"] == split, TARGET]
        axis.hist(values, bins=30, alpha=0.55, density=True, label=split)
    axis.set_xlabel("Target vehicle count (vehicles / 5 s, 300 s ahead)")
    axis.set_ylabel("Density")
    axis.set_title("Target distribution by chronological split")
    axis.legend()
    save(figure, "target_distribution_by_split.png")

    # Target by scenario and split (means with standard-deviation error bars).
    grouped = (
        data.groupby(["scenario", "split"], observed=True)[TARGET]
        .agg(["mean", "std"])
        .reset_index()
    )
    scenarios = sorted(grouped["scenario"].unique())
    positions = np.arange(len(scenarios))
    width = 0.8 / len(SPLITS)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    for index, split in enumerate(SPLITS):
        subset = grouped[grouped["split"] == split].set_index("scenario")
        axis.bar(
            positions + (index - (len(SPLITS) - 1) / 2) * width,
            [subset.loc[name, "mean"] if name in subset.index else np.nan for name in scenarios],
            width,
            yerr=[subset.loc[name, "std"] if name in subset.index else np.nan for name in scenarios],
            capsize=3,
            label=split,
        )
    axis.set_xticks(positions, scenarios)
    axis.set_ylabel("Mean target vehicle count")
    axis.set_title("Target by scenario and split")
    axis.legend()
    save(figure, "target_by_scenario_split.png")

    # Target by direction.
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.boxplot(
        [data.loc[data["direction"] == direction, TARGET] for direction in DIRECTIONS],
        tick_labels=list(DIRECTIONS),
        showfliers=False,
    )
    axis.set_ylabel("Target vehicle count")
    axis.set_title("Target distribution by approach direction")
    save(figure, "target_by_direction.png")

    # Top feature-target correlations.
    top = correlations.head(15).iloc[::-1]
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.barh(top["feature"], top["corr_all"], color="#4472c4")
    axis.set_xlabel("Pearson correlation with target")
    axis.set_title("Top 15 feature-target correlations")
    axis.axvline(0, color="black", linewidth=0.8)
    save(figure, "top_correlations.png")

    # Largest train-vs-validation/test distribution shifts.
    top_shifts = shifts.head(15).iloc[::-1]
    figure, axis = plt.subplots(figsize=(8, 5.5))
    positions = np.arange(len(top_shifts))
    axis.barh(positions - 0.2, top_shifts["smd_train_validation"], 0.4, label="train vs validation")
    axis.barh(positions + 0.2, top_shifts["smd_train_test"], 0.4, label="train vs test")
    axis.set_yticks(positions, top_shifts["column"])
    axis.set_xlabel("Standardised mean difference")
    axis.set_title("Largest split distribution shifts")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.legend()
    save(figure, "split_shift.png")

    # One run's demand over time, with split boundaries.
    available_runs = sorted(data["run_id"].unique())
    selected_run = run_id if run_id in available_runs else available_runs[0]
    run_data = data[data["run_id"] == selected_run]
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for direction in DIRECTIONS:
        subset = run_data[run_data["direction"] == direction].sort_values("simulation_time")
        axis.plot(subset["simulation_time"], subset["vehicle_count"], label=direction, linewidth=1.2)
    for split, colour in zip(SPLITS, ("#70ad47", "#ed7d31", "#c00000")):
        subset = run_data[run_data["split"] == split]
        if not subset.empty:
            axis.axvspan(
                subset["simulation_time"].min(),
                subset["simulation_time"].max(),
                color=colour,
                alpha=0.10,
                label=split,
            )
    axis.set_xlabel("Simulation time (s)")
    axis.set_ylabel("Vehicle count per 5 s")
    axis.set_title(f"Demand over time: {selected_run}")
    axis.legend(ncol=2, fontsize=8)
    save(figure, "demand_timeseries.png")

    # Current count history versus the 5-minute-ahead target.
    sample = data.sample(n=min(6000, len(data)), random_state=42)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(sample["count_lag_60"], sample[TARGET], s=5, alpha=0.25)
    axis.set_xlabel("Vehicle count 300 s ago (count_lag_60)")
    axis.set_ylabel("Target vehicle count (300 s ahead)")
    axis.set_title("Five-minute persistence relationship")
    save(figure, "count_lag_60_vs_target.png")

    # Target by signal phase.
    phase_means = data.groupby("traffic_light_phase", observed=True)[TARGET].mean()
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(phase_means.index.astype(str), phase_means.values, color="#4472c4")
    axis.set_xlabel("Traffic-light phase index")
    axis.set_ylabel("Mean target vehicle count")
    axis.set_title("Target by active signal phase")
    save(figure, "target_by_phase.png")

    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--feature-map", type=Path, default=DEFAULT_FEATURE_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SETS,
        default="F4",
        help="feature set to describe and correlate (default: F4)",
    )
    parser.add_argument("--run-id", help="run to plot in demand_timeseries.png")
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    feature_map = load_feature_map(args.feature_map)
    features = feature_map[args.feature_set]
    lookup = feature_set_lookup(feature_map)

    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        raise ValueError(
            f"Dataset is missing features from {args.feature_set}: {', '.join(missing_features)}"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    overview = build_overview(data, feature_map)
    target_by_split = distribution_summary(data, ["split"])
    target_by_scenario = distribution_summary(data, ["scenario"])
    target_by_direction = distribution_summary(data, ["direction"])
    target_by_scenario_direction = distribution_summary(data, ["scenario", "direction"])
    target_by_scenario_split = distribution_summary(data, ["scenario", "split"])
    features_table = feature_summary(data, features, lookup)
    correlations = correlation_table(data, features, lookup)
    autocorrelations = autocorrelation_table(data)
    shifts = split_shift_table(data, features, lookup)
    phases = phase_summary(data)

    tables = {
        "target_by_split.csv": target_by_split,
        "target_by_scenario.csv": target_by_scenario,
        "target_by_direction.csv": target_by_direction,
        "target_by_scenario_direction.csv": target_by_scenario_direction,
        "target_by_scenario_split.csv": target_by_scenario_split,
        "feature_summary.csv": features_table,
        "correlations.csv": correlations,
        "autocorrelation.csv": autocorrelations,
        "split_shift.csv": shifts,
        "target_by_phase.csv": phases,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False)

    plots = make_plots(data, correlations, shifts, output_dir, run_id=args.run_id)

    summary = {
        "overview": overview,
        "target_by_split": target_by_split.round(4).to_dict(orient="records"),
        "target_by_scenario_split": target_by_scenario_split.round(4).to_dict(orient="records"),
        "top_correlations": correlations.head(10).round(4).to_dict(orient="records"),
        "largest_split_shifts": shifts.head(10).round(4).to_dict(orient="records"),
        "autocorrelation_lag_60": autocorrelations[
            (autocorrelations["lag_samples"] == 60) & (autocorrelations["scope"] == "all")
        ].round(4).to_dict(orient="records"),
        "tables": sorted(tables),
        "plots": [path.name for path in plots],
    }
    (output_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Rows: {overview['rows']:,} | runs: {overview['runs']} | missing cells: {overview['missing_cells']}")
    print("Target mean by split:")
    for row in target_by_split.itertuples(index=False):
        print(f"  {row.split:<11} n={row.rows:>6,}  mean={row.mean:6.3f}  std={row.std:6.3f}")
    print("Top feature-target correlations:")
    for row in correlations.head(5).itertuples(index=False):
        print(f"  {row.feature:<24} r={row.corr_all: .3f}")
    print("Largest split shifts (standardised mean difference vs train):")
    for row in shifts.head(5).itertuples(index=False):
        print(
            f"  {row.column:<24} validation={row.smd_train_validation: .3f}"
            f"  test={row.smd_train_test: .3f}"
        )
    print(f"Wrote {len(tables)} tables and {len(plots)} plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
