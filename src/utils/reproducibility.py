"""Deterministic seeding helpers.

Every training or evaluation entry point calls :func:`set_global_seed` once
before building models or environments so that runs are repeatable across
the SL and RL lanes.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> int:
    """Seed Python, NumPy and (when installed) torch RNGs.

    Returns the seed so callers can record it in metrics artefacts. torch is
    optional: the SL lane and Q-Learning must work without it.
    """
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CPU-only environments
        torch.cuda.manual_seed_all(seed)
    return seed
