"""Tests for SUMO binary resolution helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.simulation.sumo_binary import resolve_sumo_binary


def test_absolute_binary_path_is_returned() -> None:
    path = str((Path.cwd() / "fake-sumo.exe").resolve())
    assert resolve_sumo_binary(path) == path


def test_missing_binary_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUMO_HOME", raising=False)

    with pytest.raises(FileNotFoundError, match="SUMO_HOME"):
        resolve_sumo_binary("definitely-not-a-sumo-binary")
