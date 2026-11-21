"""Tests for the shared SL/RL training and evaluation harnesses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reinforcement.agents import available_agents, make_agent, register_agent
from src.reinforcement.environment import GREEN_PHASES, Observation
from src.reinforcement.train import resolve_config
from src.supervised import load_splits
from src.supervised.evaluate import evaluate_predictor
from src.supervised.train import train_models
from src.utils import deep_merge, set_global_seed


class DummyAgent:
    """Minimal agent used only to prove the harness runs any registered agent."""

    name = "dummy"

    def __init__(self, config=None):
        self.config = dict(config or {})
        self.epsilon = 0.0

    def reset(self) -> None:
        return None

    def act(self, observation: Observation) -> int:
        return GREEN_PHASES[0]

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        target = path / "checkpoint.npz"
        target.write_bytes(b"dummy")
        return target


def sample_observation(phase: int = 0) -> Observation:
    return Observation(
        simulation_time=10.0,
        phase=phase,
        time_in_phase=5.0,
        phase_duration=15.0,
        counts=(4, 3, 2, 1),
        queues=(2, 1, 0, 1),
        speeds=(10.0, 11.0, 12.0, 13.0),
        waiting=(20.0, 10.0, 0.0, 5.0),
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def test_set_global_seed_is_deterministic() -> None:
    set_global_seed(7)
    first = np.random.rand(3)
    set_global_seed(7)
    second = np.random.rand(3)
    assert np.array_equal(first, second)


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"x": 1}, "b": 2}
    override = {"a": {"y": 3}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 3}, "b": 2}
    assert base == {"a": {"x": 1}, "b": 2}
    assert override == {"a": {"y": 3}}


# ---------------------------------------------------------------------------
# SL harness
# ---------------------------------------------------------------------------
def test_train_models_writes_validation_metrics(tmp_path) -> None:
    from src.preprocessing.features import load_sl_config

    config = load_sl_config("configs/sl.yaml")
    records = train_models(
        config=config,
        feature_set="F0",
        model_names=["persistence", "ridge"],
        seed=42,
        output_dir=tmp_path / "metrics",
        save_dir=None,
        save=False,
    )
    assert [record["model"] for record in records] == ["persistence", "ridge"]
    for record in records:
        assert record["n_validation"] == 9600
        assert set(record["validation"]) == {"mae", "rmse", "r2", "mape"}
        # Test split must never be touched during training.
        assert "test" not in record

    metrics_file = tmp_path / "metrics" / "F0" / "ridge.json"
    assert metrics_file.is_file()
    saved = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert saved["validation"]["mae"] == records[1]["validation"]["mae"]
    assert (tmp_path / "metrics" / "F0" / "summary.csv").is_file()


def test_evaluate_predictor_scores_a_split(tmp_path) -> None:
    from src.preprocessing.features import load_sl_config
    from src.supervised import fit_named_predictor

    config = load_sl_config("configs/sl.yaml")
    splits = load_splits(config, feature_set="F0")
    predictor = fit_named_predictor(
        "ridge", splits["train"].X, splits["train"].y, metadata={"feature_set": "F0"}
    )
    record = evaluate_predictor(
        predictor,
        config=config,
        feature_set="F0",
        model_name="ridge",
        split="validation",
        output_dir=tmp_path / "metrics",
    )
    assert record["n_rows"] == 9600
    assert record["metrics"]["mae"] > 0
    assert (tmp_path / "metrics" / "F0" / "ridge_validation.json").is_file()


def test_persistence_predicts_through_a_fitted_pipeline() -> None:
    """Regression test: PersistenceRegressor must satisfy the sklearn
    estimator protocol so ``Pipeline.predict`` works (it previously raised
    ``AttributeError: ... has no attribute '__sklearn_tags__'``)."""
    from src.preprocessing.features import load_sl_config
    from src.supervised import fit_named_predictor

    config = load_sl_config("configs/sl.yaml")
    splits = load_splits(config, feature_set="F0")
    predictor = fit_named_predictor(
        "persistence", splits["train"].X, splits["train"].y
    )
    # Direct call: no workaround helper may be required.
    predictions = predictor.pipeline.predict(splits["validation"].X)
    assert len(predictions) == len(splits["validation"].X)
    expected = splits["validation"].X["vehicle_count"].to_numpy(dtype=float)
    assert np.allclose(predictions, expected)


def test_time_block_folds_use_a_300_second_embargo() -> None:
    """SL-3: folds cut on simulation_time, all runs in every fold."""
    from src.supervised.train import time_block_folds

    rows = [
        {"run_id": run, "simulation_time": time_s}
        for run in ("run_a", "run_b")
        for time_s in range(305, 1805, 5)
    ]
    frame = pd.DataFrame(rows)

    folds = time_block_folds(frame, n_splits=3)
    assert len(folds) == 3
    for (train_idx, val_idx), train_end in zip(
        folds, (600, 900, 1200), strict=True
    ):
        train_times = frame.iloc[train_idx]["simulation_time"]
        val_times = frame.iloc[val_idx]["simulation_time"]
        assert train_times.max() == train_end
        assert val_times.min() == train_end + 305  # >= 300 s embargo
        assert val_times.max() - val_times.min() == 295  # 60 instants = 300 s
        assert set(frame.iloc[train_idx]["run_id"]) == {"run_a", "run_b"}
        assert set(frame.iloc[val_idx]["run_id"]) == {"run_a", "run_b"}

    with pytest.raises(ValueError):
        time_block_folds(frame, n_splits=99)


def test_effective_params_passes_the_seed_to_seeded_models() -> None:
    """SL-2: --seed reaches random_state; persistence has none."""
    from src.supervised.train import _effective_params

    assert _effective_params("decision_tree", {}, seed=43)["random_state"] == 43
    ridge = _effective_params(
        "ridge", {"models": {"ridge": {"params": {"alpha": 2.0}}}}, seed=7
    )
    assert ridge == {"alpha": 2.0, "random_state": 7}
    assert "random_state" not in _effective_params("persistence", {}, seed=7)


# ---------------------------------------------------------------------------
# RL agent registry (agents ship per-lane; the harness must work with any)
# ---------------------------------------------------------------------------
def test_registry_rejects_unknown_names() -> None:
    # NOTE: the registry is intentionally NOT asserted empty here — lanes
    # register their agents in this shared module (contract-v2 fix).
    with pytest.raises(KeyError):
        make_agent("not_an_agent", {})


def test_registry_registers_and_builds_an_agent() -> None:
    register_agent(DummyAgent.name, DummyAgent)
    try:
        assert "dummy" in available_agents()
        agent = make_agent("dummy", {"state": "S0"})
        assert isinstance(agent, DummyAgent)
        assert agent.config["state"] == "S0"
        assert agent.act(sample_observation()) in GREEN_PHASES
    finally:
        from src.reinforcement.agents import _AGENTS

        _AGENTS.pop("dummy", None)


def test_resolve_config_prefers_agent_block() -> None:
    rl_config = {"training": {"gamma": 0.95, "episodes": 200}}
    agents_config = {
        "training": {"save_every": 10},
        "dummy": {"alpha": 0.3, "gamma": 0.9},
    }
    merged = resolve_config(rl_config, agents_config, "dummy")
    assert merged["alpha"] == 0.3
    assert merged["gamma"] == 0.9  # agent block wins over rl.yaml
    assert merged["episodes"] == 200
    assert merged["save_every"] == 10


# ---------------------------------------------------------------------------
# RL harness integration (real SUMO, kept short)
# ---------------------------------------------------------------------------
def test_rl_train_writes_log_and_checkpoint(tmp_path) -> None:
    from src.reinforcement.train import train
    from src.utils import load_yaml

    register_agent(DummyAgent.name, DummyAgent)
    try:
        summary = train(
            agent_name="dummy",
            rl_config=load_yaml("configs/rl.yaml"),
            agents_config=load_yaml("configs/rl_agents.yaml"),
            scenario="low",
            seed=42,
            episodes=1,
            duration=60,
            output_dir=tmp_path / "results",
            save_dir=tmp_path / "models",
        )
    finally:
        from src.reinforcement.agents import _AGENTS

        _AGENTS.pop("dummy", None)
    log_path = Path(summary["log"])
    assert log_path.is_file()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # header + one episode
    assert summary["checkpoint"].endswith("checkpoint.npz")
    assert (tmp_path / "models").is_dir()


def test_rl_evaluate_fixed_baseline(tmp_path) -> None:
    from src.reinforcement.evaluate import evaluate
    from src.utils import load_yaml

    summary = evaluate(
        agent_name="fixed",
        rl_config=load_yaml("configs/rl.yaml"),
        scenarios=["low"],
        seeds=[42],
        episodes=1,
        duration=60,
        output_dir=tmp_path / "results",
    )
    assert summary["n_rows"] == 1
    assert summary["metrics"]["mean_waiting_total"]["n"] == 1
    assert (tmp_path / "results" / "fixed").is_dir()
