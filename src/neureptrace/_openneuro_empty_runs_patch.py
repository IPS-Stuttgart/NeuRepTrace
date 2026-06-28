"""Reject empty OpenNeuro run selections before staging."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from functools import wraps

_OPENNEURO_EMPTY_RUNS_PATCH_MARKER = "_neureptrace_openneuro_empty_runs_patch_installed"
_EMPTY_RUN_SELECTION_MESSAGE = "OpenNeuro run selection must include at least one run; use 'all' or pass one or more run ids."


def install() -> None:
    """Patch OpenNeuro run parsing to reject empty run selections."""

    openneuro_meg = importlib.import_module("neureptrace.openneuro_meg")
    if getattr(openneuro_meg, _OPENNEURO_EMPTY_RUNS_PATCH_MARKER, False):
        return

    original_parse_runs = openneuro_meg.parse_runs

    @wraps(original_parse_runs)
    def parse_runs(spec, runs: str | Iterable[str] | None):
        selected_runs = tuple(original_parse_runs(spec, runs))
        if not selected_runs:
            raise ValueError(_EMPTY_RUN_SELECTION_MESSAGE)
        return selected_runs

    openneuro_meg.parse_runs = parse_runs
    setattr(openneuro_meg, _OPENNEURO_EMPTY_RUNS_PATCH_MARKER, True)


__all__ = ["install"]
