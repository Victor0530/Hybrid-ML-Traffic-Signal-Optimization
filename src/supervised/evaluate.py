"""Shared supervised-learning evaluation entry point.

Loads a saved predictor and scores it on one frozen split. The default split
is ``validation``; ``test`` must be requested explicitly because the contract
requires touching the test split exactly once, after model selection.

Usage:
    python -m src.supervised.evaluate --model ridge --feature-set F4
    python -m src.supervised.evaluate --model ridge --split test
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.metrics import regression_metrics
from src.preprocessing.features import load_sl_config
from src.supervised import Predictor, load_splits
from src.utils import set_global_seed

DEFAULT_CONFIG = Path("configs/sl.yaml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", required=True, help="registered model name (metrics record key)")
    parser.add_argument("--feature-set", default=None, help="default: config features.default_set")
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
        help="evaluation split; test is final-evaluation only",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predictor", type=Path, default=None, help="default: models/sl/<feature_set>/<model>/predictor.joblib")
    parser.add_argument("--output-dir", type=Path, default=Path("results/sl/metrics"))
    return parser.parse_args(argv)


def evaluate_predictor(
    predictor: Predictor,
    config: dict[str, Any],
    feature_set: str,
    model_name: str,
    split: str = "validation",
    seed: int = 42,
    output_dir: Path = Path("results/sl/metrics"),
) -> dict[str, Any]:
    """Score one saved predictor on one frozen split and persist the record."""
    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unknown split {split!r}")
    splits = load_splits(config, feature_set=feature_set)
    data = splits[split]
    predictions = predictor.pipeline.predict(data.X)
    record = {
        "model": model_name,
        "feature_set": feature_set,
        "split": split,
        "seed": seed,
        "n_rows": int(len(data.X)),
        "metrics": regression_metrics(data.y, predictions),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path(output_dir) / feature_set
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{model_name}_{split}.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return record


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_global_seed(args.seed)
    config = load_sl_config(args.config)
    feature_set = args.feature_set or config["features"]["default_set"]

    predictor_path = args.predictor or (
        Path("models/sl") / feature_set / args.model / "predictor.joblib"
    )
    if not predictor_path.is_file():
        raise FileNotFoundError(
            f"No saved predictor at {predictor_path}; run "
            f"`python -m src.supervised.train --models {args.model}` first"
        )
    if args.split == "test":
        print(
            "WARNING: evaluating on the TEST split. Run this only once, "
            "after the final model has been selected on validation."
        )

    predictor = Predictor.load(predictor_path)
    record = evaluate_predictor(
        predictor,
        config=config,
        feature_set=feature_set,
        model_name=args.model,
        split=args.split,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    metrics = record["metrics"]
    print(
        f"{args.model} on {args.split} ({feature_set}, {record['n_rows']} rows): "
        f"MAE {metrics['mae']:.4f}  RMSE {metrics['rmse']:.4f}  "
        f"R2 {metrics['r2']:.4f}  MAPE {metrics['mape']:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
