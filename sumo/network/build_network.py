"""Build intersection.net.xml from the editable plain-XML network sources.

Run from anywhere with:
    python sumo/network/build_network.py

Requires Eclipse SUMO's netconvert. The script prefers PATH and falls back to
%SUMO_HOME%/bin, so it also works from a terminal with a stale environment.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.sumo_binary import resolve_sumo_binary

HERE = Path(__file__).resolve().parent


def main() -> int:
    try:
        netconvert = resolve_sumo_binary("netconvert")
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Relative paths keep the generated net.xml portable between machines.
    command = [
        netconvert,
        "--node-files", "intersection.nod.xml",
        "--edge-files", "intersection.edg.xml",
        "--connection-files", "intersection.con.xml",
        "--output-file", "intersection.net.xml",
        "--no-turnarounds", "true",
        "--offset.disable-normalization", "true",
    ]
    print("Running:", " ".join(command), f"(cwd={HERE})")
    subprocess.run(command, cwd=HERE, check=True)
    print(f"Built {HERE / 'intersection.net.xml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())