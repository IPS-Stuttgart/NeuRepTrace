"""Reject boolean-like numeric hyperparameters for linear VREx."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_vrex_numeric_config_patch_installed"


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative.") from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return parsed


def install() -> None:
    """Install linear VREx numeric hyperparameter validators."""

    vrex = importlib.import_module("neureptrace.decoding.vrex")
    if getattr(vrex._positive_int, _PATCH_MARKER, False):
        return

    setattr(_positive_int, _PATCH_MARKER, True)
    setattr(_positive_float, _PATCH_MARKER, True)
    setattr(_nonnegative_float, _PATCH_MARKER, True)
    vrex._positive_int = _positive_int
    vrex._positive_float = _positive_float
    vrex._nonnegative_float = _nonnegative_float


__all__ = ["install"]
