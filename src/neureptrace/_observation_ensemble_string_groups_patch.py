"""Normalize string baseline grouping columns for observation ensembles."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

_PATCH_MARKER = "_neureptrace_observation_ensemble_string_groups_patch_installed"


def _normalize_baseline_group_columns(columns: Sequence[str] | str | None) -> tuple[str, ...]:
    """Return baseline group columns without treating a single string as characters."""

    if columns is None:
        return ()
    if isinstance(columns, str):
        return (columns,)
    return tuple(dict.fromkeys(columns))


def install() -> None:
    """Patch observation ensembling to accept one-column baseline groups as strings."""

    observation_ensemble = importlib.import_module("neureptrace.observation_ensemble")
    original = observation_ensemble.ensemble_probability_observations
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def ensemble_probability_observations(*args: Any, **kwargs: Any):
        """Call the original implementation with normalized baseline grouping columns."""

        if "baseline_group_columns" in kwargs:
            kwargs = dict(kwargs)
            kwargs["baseline_group_columns"] = _normalize_baseline_group_columns(kwargs["baseline_group_columns"])
        return original(*args, **kwargs)

    setattr(ensemble_probability_observations, _PATCH_MARKER, True)
    observation_ensemble.ensemble_probability_observations = ensemble_probability_observations


__all__ = ["install"]
