"""Agent registry shared by the RL training and evaluation entry points.

This is an **extension point, not an implementation**. The project ships no
trained or hand-written agent: each lane adds its own module under
``src/reinforcement/`` and registers a factory here with two lines::

    from src.reinforcement.agents import register_agent
    from src.reinforcement.my_agent import MyAgent

    register_agent(MyAgent.name, MyAgent)

After that ``python -m src.reinforcement.train --agent my_agent`` and
``python -m src.reinforcement.evaluate --agent my_agent`` work unchanged.

An agent factory is any callable taking one mapping of hyper-parameters and
returning an object with the interface used by the harness:

- ``reset()``                 -- start of an episode
- ``act(observation)``        -- green phase from ``GREEN_PHASES``
- ``save(path)``              -- write a checkpoint, return its path
- ``epsilon`` (optional)      -- reported in the training log; the evaluator
                                 forces it to 0 for greedy runs
- ``explore`` (optional)      -- bool flag; the evaluator forces it to False
- classmethod ``load(path)``  -- used by the evaluator
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

AgentFactory = Callable[[Mapping[str, Any]], Any]

_AGENTS: dict[str, AgentFactory] = {}


def register_agent(name: str, factory: AgentFactory) -> None:
    """Register or replace an agent factory under ``name``."""
    key = name.strip().lower()
    if not key:
        raise ValueError("Agent name must not be empty")
    _AGENTS[key] = factory


def available_agents() -> tuple[str, ...]:
    """Return every registered agent name."""
    return tuple(sorted(_AGENTS))


def make_agent(name: str, config: Mapping[str, Any] | None = None) -> Any:
    """Build an agent by name with its merged configuration block."""
    key = name.strip().lower()
    if key not in _AGENTS:
        raise KeyError(f"Unknown agent {name!r}; choose from {available_agents()}")
    return _AGENTS[key](config or {})
