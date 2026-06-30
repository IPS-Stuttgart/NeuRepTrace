"""Normalize source-interpolation seed and scalar numeric controls."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_INSTALLED = False
_PATCH_MARKER = "_neureptrace_source_interpolation_seed_config_patch_installed"


def _numeric_scalar(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a numeric scalar, not a boolean.")
    if isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a numeric scalar, not a NumPy array.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric scalar.") from exc


def _nonnegative_int(value: Any, *, name: str) -> int:
    parsed = _numeric_scalar(value, name=name)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 0.0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(parsed)


def _positive_float(value: Any, *, name: str) -> float:
    parsed = _numeric_scalar(value, name=name)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return float(parsed)


def _unit_interval_float(value: Any, *, name: str) -> float:
    parsed = _numeric_scalar(value, name=name)
    if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return float(parsed)


def _optional_random_state(value: Any, *, name: str = "random_state") -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return _nonnegative_int(value, name=name)


def install() -> None:
    """Patch source-interpolation config normalization."""

    global _INSTALLED
    if _INSTALLED:
        return

    module = importlib.import_module("neureptrace.decoding.source_interpolation")
    original_config = module.source_interpolation_config
    if getattr(original_config, _PATCH_MARKER, False):
        _INSTALLED = True
        return

    @wraps(original_config)
    def source_interpolation_config(*args: Any, **kwargs: Any):
        if "random_state" in kwargs:
            kwargs = dict(kwargs)
            kwargs["random_state"] = _optional_random_state(kwargs["random_state"], name="random_state")
        return original_config(*args, **kwargs)

    setattr(source_interpolation_config, _PATCH_MARKER, True)
    module._nonnegative_int = _nonnegative_int
    module._positive_float = _positive_float
    module._unit_interval_float = _unit_interval_float
    module.source_interpolation_config = source_interpolation_config
    _INSTALLED = True


__all__ = ["install"]
