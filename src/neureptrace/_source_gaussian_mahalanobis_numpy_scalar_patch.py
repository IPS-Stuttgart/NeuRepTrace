"""Accept scalar NumPy numeric config controls for Gaussian/Mahalanobis source decoders."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

_INSTALLED = False


def _numeric_scalar(value: Any, *, name: str, allow_zero: bool) -> float:
    kind = "non-negative" if allow_zero else "positive"
    message = f"{name} must be {kind} and finite."
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        return _numeric_scalar(value.item(), name=name, allow_zero=allow_zero)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(parsed) or parsed < 0.0 or (not allow_zero and parsed <= 0.0):
        raise ValueError(message)
    return parsed


def _positive_float(value: Any, *, name: str) -> float:
    return _numeric_scalar(value, name=name, allow_zero=False)


def _nonnegative_float(value: Any, *, name: str) -> float:
    return _numeric_scalar(value, name=name, allow_zero=True)


def install() -> None:
    """Install scalar NumPy numeric config normalization for source decoder helpers."""

    global _INSTALLED
    if _INSTALLED:
        return

    importlib.import_module("neureptrace._source_interpolation_one_pass_patch").install()

    from neureptrace.decoding import source_gaussian, source_mahalanobis

    source_gaussian._positive_float = _positive_float
    source_mahalanobis._positive_float = _positive_float
    source_mahalanobis._nonnegative_float = _nonnegative_float
    _INSTALLED = True


__all__ = ["install"]
