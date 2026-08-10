"""Normalize and validate observation-ensemble scalar controls."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_observation_ensemble_source_debias_bool_patch_installed"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}
_REAL_CONTROL_NAMES = (
    "weights",
    "source_temperatures",
    "probability_tolerance",
    "baseline_window",
)


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


def _contains_complex(value: Any) -> bool:
    """Return whether a scalar or small control container contains complex values."""

    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return True
        if value.dtype == object:
            return any(_contains_complex(item) for item in value.flat)
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_complex(item) for item in value)
    to_numpy = getattr(value, "to_numpy", None)
    if callable(to_numpy):
        return _contains_complex(to_numpy())
    return False


def _validate_real_control(value: Any, *, name: str) -> None:
    if value is not None and _contains_complex(value):
        raise ValueError(f"{name} must contain only real-valued numbers.")


def install() -> None:
    """Patch observation-ensemble boolean normalization and real-control validation."""

    module = importlib.import_module("neureptrace.observation_ensemble")
    if getattr(module, _PATCH_MARKER, False):
        return

    original_ensemble_probability_observations = module.ensemble_probability_observations

    @wraps(original_ensemble_probability_observations)
    def ensemble_probability_observations(*args: Any, **kwargs: Any):
        if "source_baseline_debiasing" in kwargs:
            kwargs = dict(kwargs)
            kwargs["source_baseline_debiasing"] = normalize_source_baseline_debiasing(kwargs["source_baseline_debiasing"])
        for name in _REAL_CONTROL_NAMES:
            if name in kwargs:
                _validate_real_control(kwargs[name], name=name)
        return original_ensemble_probability_observations(*args, **kwargs)

    setattr(ensemble_probability_observations, _PATCH_MARKER, True)
    module.ensemble_probability_observations = ensemble_probability_observations
    setattr(module, _PATCH_MARKER, True)


__all__ = ["install", "normalize_source_baseline_debiasing"]
