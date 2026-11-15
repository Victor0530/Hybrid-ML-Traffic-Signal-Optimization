"""Build the supervised-learning dataset, splits, feature map and audit.

Usage:
    python -m src.preprocessing.build_dataset
    python -m src.preprocessing.build_dataset --config configs/sl.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing.features import (
    DEFAULT_CONFIG,
    FEATURE_SET_COLUMNS,
    PROJECT_ROOT,
    build_dataset,
    load_sl_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the SL configuration YAML (default: configs/sl.yaml).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = load_sl_config(config_path)

    dataset, audit = build_dataset(config)
    if dataset.empty:
        raise SystemExit("No modelling rows were produced; check the raw dataset.")

    split_rows = audit["split_rows"]
    if any(split_rows.get(name, 0) == 0 for name in ("train", "validation", "test")):
        raise SystemExit(f"Empty split detected: {split_rows}")

    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    print(f"Runs loaded:      {audit['runs']}")
    print(f"Raw rows:         {audit['raw_rows']:,}")
    print(f"Dataset rows:     {audit['dataset_rows']:,}")
    print(f"Target dropped:   {audit['target_undefined_rows']:,}")
    print(f"Warm-up dropped:  {audit['warmup_rows_dropped']:,}")
    print(f"Embargo excluded: {audit['embargo_excluded_rows']:,}")
    print(
        "Splits:           "
        f"train={split_rows['train']:,}, "
        f"validation={split_rows['validation']:,}, "
        f"test={split_rows['test']:,}"
    )
    print(
        "Feature counts:   "
        + ", ".join(
            f"{name}={len(columns)}" for name, columns in FEATURE_SET_COLUMNS.items()
        )
    )
    print(f"Wrote dataset:    {processed_dir / config['output']['dataset']}")
    print(f"Wrote splits:     {processed_dir / Path(config['data']['splits_dir']).name}")
    print(f"Wrote feature map:{processed_dir / config['output']['feature_map']}")
    print(f"Wrote audit:      {processed_dir / config['output']['audit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
