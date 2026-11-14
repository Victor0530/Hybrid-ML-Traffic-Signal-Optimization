"""Locate SUMO executables even when the terminal's PATH is stale."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path


def resolve_sumo_binary(name: str) -> str:
    """Return an absolute path to a SUMO binary.

    Prefers PATH, then falls back to %SUMO_HOME%/bin. This matters on Windows
    when the terminal was opened before SUMO was installed.

    Raises:
        FileNotFoundError: if the executable cannot be found.
    """
    if os.path.isabs(name):
        return name

    found = shutil.which(name)
    if found is not None:
        return found

    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        executable = name
        if os.name == "nt" and not executable.lower().endswith(".exe"):
            executable = f"{executable}.exe"
        candidate = Path(sumo_home) / "bin" / executable
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f"Could not find SUMO binary {name!r}. Set SUMO_HOME to the SUMO "
        "installation directory (for example C:\\Program Files (x86)\\SUMO) "
        "or add its bin folder to PATH."
    )


def get_sumo_version(binary: str) -> str:
    """Return the version reported by the SUMO binary itself, or an empty string."""
    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""

    output = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"SUMO\s+\S+\s+(\d+\.\d+\.\d+)", output)
    if match is None:
        match = re.search(r"\b(\d+\.\d+\.\d+)\b", output)
    return match.group(1) if match else ""