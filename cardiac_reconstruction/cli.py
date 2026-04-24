"""Command line interface for the project."""

from __future__ import annotations

import argparse
from typing import Callable, Dict

from .entrypoints import (
    inspect_dataset,
    run_ablation,
    run_all,
    run_baseline,
    run_complete_pipeline,
    run_sparse_reconstruction,
    run_viewer,
)

COMMANDS: Dict[str, Callable[[], None]] = {
    "inspect": inspect_dataset,
    "baseline": run_baseline,
    "ablation": run_ablation,
    "sparse": run_sparse_reconstruction,
    "viewer": run_viewer,
    "complete": run_complete_pipeline,
    "all": run_all,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cardiac_reconstruction",
        description="Canonical command surface for the cardiac reconstruction project.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=sorted(COMMANDS),
        help="Workflow to run (default: all).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    COMMANDS[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
