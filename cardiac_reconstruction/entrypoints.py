"""Command wrappers for the canonical project workflows."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def _run(script_name: str) -> None:
    script_path = SRC_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find script: {script_path}")

    subprocess.run([sys.executable, str(script_path)], check=True, cwd=PROJECT_ROOT)


def inspect_dataset() -> None:
    _run("data_inspector.py")


def run_baseline() -> None:
    _run("minimal_starter_5.py")


def run_ablation() -> None:
    _run("ablation_studies_7.py")


def run_sparse_reconstruction() -> None:
    _run("sparse_reconstruction_2.py")


def run_viewer() -> None:
    _run("viewer_3d_reconstruction_2.py")


def run_complete_pipeline() -> None:
    _run("complete_pipeline.py")


def run_all() -> None:
    _run("run_all.py")
