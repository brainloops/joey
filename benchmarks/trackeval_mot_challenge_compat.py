#!/usr/bin/env python3

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np


def apply_numpy_compat_aliases() -> None:
    # TrackEval still references legacy numpy scalar aliases removed in numpy>=2.
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]


def main() -> None:
    apply_numpy_compat_aliases()
    root_dir = Path(__file__).resolve().parent.parent
    target = root_dir / "benchmarks/repos/TrackEval/scripts/run_mot_challenge.py"
    if not target.is_file():
        raise FileNotFoundError(f"TrackEval runner not found: {target}")
    sys.argv = [str(target), *sys.argv[1:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
