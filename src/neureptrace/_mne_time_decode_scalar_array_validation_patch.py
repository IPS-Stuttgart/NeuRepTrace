"""Reject NumPy array-valued scalar controls in MNE time decoding."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_mne_time_decode_scalar_array_validation_patch_installed"


def _is_array_scalar_control(value: Any) -> bool:
    """Return true for ndarray values that NumPy may coerce to scalar controls."""

    return isinstance(value, np.ndarray)


def _integer_error(name: str) -> ValueError:
    return ValueError(f"{name} must be an integer.")


def _positive_float_error(name: str) -> ValueError:
    return ValueError(f"{name} must be positive and finite.")


def _nonnegative_float_error(name: str) -> ValueError:
    return ValueError(f"{name} must be non-negative and finite.")


def _unit_interval_error(name: str, *, include_one: bool = False) -> ValueError:
    bracket = "[0, 1]" if include_one else "[0, 1)"
    return ValueError(f"{name} must be finite in {bracket}.")


def _pseudo_label_threshold_error() -> ValueError:
    return ValueError("pseudo_label_confidence_threshold must be between 0 and 1.")


def _install_scalar_array_validation() -> None:
    module = importlib.import_module("neureptrace.mne_time_decode")
    original_normalize_integer = module._normalize_integer
    if getattr(original_normalize_integer, _PATCH_MARKER, False):
        return

    original_normalize_positive_int = module._normalize_positive_int
    original_normalize_positive_float = module._normalize_positive_float
    original_normalize_nonnegative_float = module._normalize_nonnegative_float
    original_normalize_unit_interval_float = module._normalize_unit_interval_float
    original_normalize_pseudo_label_confidence_threshold = module._normalize_pseudo_label_confidence_threshold

    @wraps(original_normalize_integer)
    def _normalize_integer(value, *, name, minimum=None):
        if _is_array_scalar_control(value):
            raise _integer_error(name)
        return original_normalize_integer(value, name=name, minimum=minimum)

    @wraps(original_normalize_positive_int)
    def _normalize_positive_int(value, *, name):
        if _is_array_scalar_control(value):
            raise _integer_error(name)
        return original_normalize_positive_int(value, name=name)

    @wraps(original_normalize_positive_float)
    def _normalize_positive_float(value, *, name):
        if _is_array_scalar_control(value):
            raise _positive_float_error(name)
        return original_normalize_positive_float(value, name=name)

    @wraps(original_normalize_nonnegative_float)
    def _normalize_nonnegative_float(value, *, name):
        if _is_array_scalar_control(value):
            raise _nonnegative_float_error(name)
        return original_normalize_nonnegative_float(value, name=name)

    @wraps(original_normalize_unit_interval_float)
    def _normalize_unit_interval_float(value, *, name, include_one=False):
        if _is_array_scalar_control(value):
            raise _unit_interval_error(name, include_one=include_one)
        return original_normalize_unit_interval_float(value, name=name, include_one=include_one)

    @wraps(original_normalize_pseudo_label_confidence_threshold)
    def _normalize_pseudo_label_confidence_threshold(value):
        if _is_array_scalar_control(value):
            raise _pseudo_label_threshold_error()
        return original_normalize_pseudo_label_confidence_threshold(value)

    setattr(_normalize_integer, _PATCH_MARKER, True)
    module._normalize_integer = _normalize_integer
    module._normalize_positive_int = _normalize_positive_int
    module._normalize_positive_float = _normalize_positive_float
    module._normalize_nonnegative_float = _normalize_nonnegative_float
    module._normalize_unit_interval_float = _normalize_unit_interval_float
    module._normalize_pseudo_label_confidence_threshold = _normalize_pseudo_label_confidence_threshold


def install() -> None:
    """Patch scalar hyperparameter parsers to reject ndarray-valued controls."""

    _install_scalar_array_validation()


__all__ = ["install"]
