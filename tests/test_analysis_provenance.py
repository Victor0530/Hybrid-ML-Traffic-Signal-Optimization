"""Tests for baseline-analysis provenance and canonical run-set checks."""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.characterise_baseline import (
    NETWORK_PATH,
    ROUTE_DIR,
    load_observations,
)
from src.simulation.data_collector import (
    hash_config,
    sha256_file,
    stable_network_hash,
)
from src.simulation.scenario_generator import load_config


def _manifest_row(config: dict, **overrides: object) -> dict:
    row = {
        "run_id": "low_seed42",
        "scenario": "low",
        "seed": 42,
        "status": "ok",
        "duration": int(config["simulation"]["duration"]),
        "sampling_interval": int(config["simulation"]["sampling_interval"]),
        "simulation_end": int(config["simulation"]["duration"]),
        "config_hash": hash_config(config),
        "route_hash": sha256_file(ROUTE_DIR / "low.rou.xml"),
        "network_hash": stable_network_hash(NETWORK_PATH),
    }
    row.update(overrides)
    return row


def _write_manifest(tmp_path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(tmp_path / "run_manifest.csv", index=False)


def test_load_observations_rejects_stale_config_hash(tmp_path) -> None:
    config = load_config()
    _write_manifest(tmp_path, [_manifest_row(config, config_hash="stale")])

    with pytest.raises(ValueError, match="different configuration"):
        load_observations(tmp_path, config)


def test_load_observations_requires_canonical_run_set(tmp_path) -> None:
    config = load_config()
    _write_manifest(tmp_path, [_manifest_row(config)])

    with pytest.raises(ValueError, match="Canonical run set mismatch"):
        load_observations(tmp_path, config)
