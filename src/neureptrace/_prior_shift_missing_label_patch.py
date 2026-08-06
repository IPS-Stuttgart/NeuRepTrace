"""Treat missing prior-shift labels as equal class and block identifiers."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Set
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

from neureptrace._object_label_utils import values_equal

_PATCH_MARKER = "_neureptrace_prior_shift_missing_label_patch_installed"


def _is_missing_label_scalar(value: Any) -> bool:
    if isinstance(value, (np.datetime64, np.timedelta64)) and bool(np.isnat(value)):
        return True
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float) and np.isnan(value)


def _comparable_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def install() -> None:
    """Make prior-shift object equality reflexive for missing labels."""

    prior_shift = importlib.import_module("neureptrace.decoding.prior_shift")
    original_object_equal = prior_shift._object_equal
    if getattr(original_object_equal, _PATCH_MARKER, False):
        return

    @wraps(original_object_equal)
    def _object_equal(left: Any, right: Any) -> bool:
        if _is_missing_label_scalar(left) or _is_missing_label_scalar(right):
            return values_equal(left, right)

        left = _comparable_scalar(left)
        right = _comparable_scalar(right)
        container_types = (np.ndarray, list, tuple, Mapping, Set)
        if isinstance(left, container_types) or isinstance(right, container_types):
            left = prior_shift._hashable_object_value(left)
            right = prior_shift._hashable_object_value(right)
            if _is_missing_label_scalar(left) or _is_missing_label_scalar(right):
                return values_equal(left, right)
            if isinstance(left, tuple) or isinstance(right, tuple):
                if not isinstance(left, tuple) or not isinstance(right, tuple) or len(left) != len(right):
                    return False
                return all(_object_equal(left_value, right_value) for left_value, right_value in zip(left, right, strict=True))

        return original_object_equal(left, right)

    setattr(_object_equal, _PATCH_MARKER, True)
    _object_equal.__wrapped__ = original_object_equal
    prior_shift._object_equal = _object_equal


__all__ = ["install"]
