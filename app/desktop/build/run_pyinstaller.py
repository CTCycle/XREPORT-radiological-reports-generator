"""Run PyInstaller with a repository-root import path for the frozen backend."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--distpath", type=Path, required=True)
    parser.add_argument("--workpath", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    app_root = repo_root / "app"
    os.environ["XREPORT_REPO_ROOT"] = str(repo_root)
    os.environ.setdefault("KERAS_BACKEND", "torch")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(app_root))
    sys.argv = [
        "pyinstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(args.distpath),
        "--workpath",
        str(args.workpath),
        str(args.spec),
    ]
    runpy.run_module("PyInstaller.__main__", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
