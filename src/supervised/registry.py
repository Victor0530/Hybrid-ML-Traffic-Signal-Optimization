"""Model registry and plugin contract.

Teammates add a model by registering a factory under a unique name. Training
and evaluation code then builds models by name, which keeps the parallel work
independent and prevents duplicated training scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.supervised.baselines import PersistenceRegressor

EstimatorFactory = Callable[..., Any]


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to construct one candidate model."""

    name: str
    factory: EstimatorFactory
    default_params: dict[str, Any] = field(default_factory=dict)
    scale: bool = False
    baseline: bool = False
    # Seeded models receive random_state=<CLI seed> at fit time so --seed
    # actually changes the fit (SL-2). Baselines such as persistence have
    # no random_state parameter.
    seeded: bool = True


_CANDIDATES: dict[str, ModelSpec] = {}
_BASELINES: dict[str, ModelSpec] = {}


def register_model(
    name: str,
    factory: EstimatorFactory,
    default_params: dict[str, Any] | None = None,
    scale: bool = False,
    baseline: bool = False,
    seeded: bool = True,
) -> None:
    """Register a model factory, replacing any existing registration."""
    key = name.strip().lower()
    if not key:
        raise ValueError("Model name must not be empty")
    target = _BASELINES if baseline else _CANDIDATES
    target[key] = ModelSpec(
        name=key,
        factory=factory,
        default_params=dict(default_params or {}),
        scale=scale,
        baseline=baseline,
        seeded=seeded,
    )


def available_models(include_baselines: bool = False) -> tuple[str, ...]:
    """Return registered candidate names, optionally including baselines."""
    names = set(_CANDIDATES)
    if include_baselines:
        names.update(_BASELINES)
    return tuple(sorted(names))


def model_spec(name: str, include_baselines: bool = True) -> ModelSpec:
    """Look up a model specification."""
    key = name.strip().lower()
    spec = _CANDIDATES.get(key)
    if spec is None and include_baselines:
        spec = _BASELINES.get(key)
    if spec is None:
        available = available_models(include_baselines=include_baselines)
        raise KeyError(f"Unknown model {name!r}; choose from {available}")
    return spec


def build_model(
    name: str,
    params: dict[str, Any] | None = None,
    include_baselines: bool = True,
) -> Any:
    """Construct an estimator, merging caller parameters over defaults."""
    spec = model_spec(name, include_baselines=include_baselines)
    merged = {**spec.default_params, **(params or {})}
    return spec.factory(**merged)


_CANDIDATES.update(
    {
        "ridge": ModelSpec(
            name="ridge",
            factory=Ridge,
            default_params={"alpha": 1.0},
            scale=True,
        ),
        "decision_tree": ModelSpec(
            name="decision_tree",
            factory=DecisionTreeRegressor,
            default_params={"random_state": 42},
        ),
        "random_forest": ModelSpec(
            name="random_forest",
            factory=RandomForestRegressor,
            default_params={"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        ),
        "hist_gradient_boosting": ModelSpec(
            name="hist_gradient_boosting",
            factory=HistGradientBoostingRegressor,
            default_params={"random_state": 42},
        ),
        "xgboost": ModelSpec(
            name="xgboost",
            factory=XGBRegressor,
            default_params={"random_state": 42, "n_jobs": -1},
        ),
    }
)

_BASELINES.update(
    {
        "persistence": ModelSpec(
            name="persistence",
            factory=PersistenceRegressor,
            baseline=True,
            seeded=False,
        )
    }
)
