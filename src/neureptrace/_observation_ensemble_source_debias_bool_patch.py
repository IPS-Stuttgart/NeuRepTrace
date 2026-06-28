"""Normalize observation-ensemble source baseline debiasing flags."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_observation_ensemble_source_debias_bool_patch_installed"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}


def _bool_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a boolean value.")


def normalize_source_baseline_debiasing(value: Any, *, name: str = "source_baseline_debiasing") -> bool:
    """Normalize YAML/CLI-style boolean tokens for source baseline debiasing."""

    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
        raise _bool_error(name)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _bool_error(name)
        return normalize_source_baseline_debiasing(value.item(), name=name)
    if isinstance(value, (int, np.integer)):
        if int(value) in {0, 1}:
            return bool(value)
        raise _bool_error(name)
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value) in {0.0, 1.0}:
            return bool(value)
        raise _bool_error(name)
    raise _bool_error(name)


def install() -> None:
    """Patch ``ensemble_probability_observations`` boolean keyword normalization."""

    module = importlib.import_module("neureptrace.observation_ensemble")
    if getattr(module, _PATCH_MARKER, False):
        return

    original_ensemble_probability_observations = module.ensemble_probability_observations

    @wraps(original_ensemble_probability_observations)
    def ensemble_probability_observations(*args: Any, **kwargs: Any):
        if "source_baseline_debiasing" in kwargs:
            kwargs = dict(kwargs)
            kwargs["source_baseline_debiasing"] = normalize_source_baseline_debiasing(kwargs["source_baseline_debiasing"])
        return original_ensemble_probability_observations(*args, **kwargs)

    setattr(ensemble_probability_observations, _PATCH_MARKER, True)
    module.ensemble_probability_observations = ensemble_probability_observations
    setattr(module, _PATCH_MARKER, True)


__all__ = ["install", "normalize_source_baseline_debiasing"]
