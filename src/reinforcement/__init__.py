"""Reinforcement-learning environment and controller contract."""

from __future__ import annotations

from typing import Any

__all__ = [
    "GREEN_PHASES",
    "Observation",
    "StepResult",
    "TrafficEnvironment",
    "observation_size",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from src.reinforcement import environment

        return getattr(environment, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
