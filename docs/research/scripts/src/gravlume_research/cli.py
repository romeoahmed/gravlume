"""Command-line boundary for the repository's Python research checks."""

import argparse
from collections.abc import Sequence
from importlib import import_module

CHECK_MODULES = {
    "bl-mino-surface": "gravlume_research.checks.bl_mino",
    "kerr-capture": "gravlume_research.checks.kerr_capture",
    "kerr-schild-map": "gravlume_research.checks.kerr_schild_map",
    "kerr-schild-rhs": "gravlume_research.checks.kerr_schild_rhs",
    "kerr-support": "gravlume_research.checks.kerr_support",
    "mino-step": "gravlume_research.checks.mino_step",
    "scalar-transport": "gravlume_research.checks.scalar_transport",
}


def _run_check(name: str) -> None:
    module = import_module(CHECK_MODULES[name])
    module.run()


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one independent Gravlume research check."
    )
    parser.add_argument("check", choices=CHECK_MODULES)
    options = parser.parse_args(arguments)
    _run_check(options.check)
    return 0
