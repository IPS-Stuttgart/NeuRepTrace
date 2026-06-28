"""Validate OpenNeuro run selectors before path expansion."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import wraps
from typing import Any

_BOOLEAN_RUN_TEXT = {"false", "no", "off", "on", "true", "yes"}
_PATCH_MARKER = "_neureptrace_openneuro_parse_runs_validation_patch_installed"


def _is_bool_like(value: Any) -> bool:
    return isinstance(value, bool) or type(value).__name__ == "bool_"


def _selector_entries(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for chunk in value.split(",") for part in chunk.split() if part.strip()]
    if isinstance(value, Iterable) and not isinstance(value, bytes):
        return list(value)
    return [value]


def _validate_run_entry(value: Any) -> None:
    if value is None:
        return
    if _is_bool_like(value):
        raise ValueError("OpenNeuro run selectors must be run identifiers, not booleans.")
    if isinstance(value, Mapping):
        raise ValueError("OpenNeuro run selectors must be scalar run identifiers, not mappings.")
    text = str(value).strip()
    if not text:
        raise ValueError("OpenNeuro run selector entries must not be empty.")
    if text.lower() in _BOOLEAN_RUN_TEXT:
        raise ValueError("OpenNeuro run selectors must be run identifiers, not booleans.")


def _normalize_runs_for_original(runs: Any) -> Any:
    if runs is None or isinstance(runs, str):
        return runs
    if isinstance(runs, Mapping):
        return runs
    if isinstance(runs, Iterable) and not isinstance(runs, bytes):
        return runs
    return (runs,)


def install() -> None:
    """Reject empty and boolean-like OpenNeuro run selectors."""

    import neureptrace.openneuro_meg as openneuro_meg

    original_parse_runs = openneuro_meg.parse_runs
    if getattr(original_parse_runs, _PATCH_MARKER, False):
        return

    @wraps(original_parse_runs)
    def parse_runs(spec: Any, runs: Any) -> tuple[str | None, ...]:
        if runs is not None:
            if isinstance(runs, Mapping):
                raise ValueError("OpenNeuro run selection must be a run id, a sequence of run ids, or 'all'.")
            if not (isinstance(runs, str) and runs.strip().lower() == "all"):
                entries = _selector_entries(runs)
                if not entries:
                    raise ValueError("OpenNeuro run selection must contain at least one run or 'all'.")
                for entry in entries:
                    _validate_run_entry(entry)

        parsed = original_parse_runs(spec, _normalize_runs_for_original(runs))
        if not parsed:
            raise ValueError("OpenNeuro run selection must contain at least one run or 'all'.")
        return parsed

    setattr(parse_runs, _PATCH_MARKER, True)
    openneuro_meg.parse_runs = parse_runs


__all__ = ["install"]
