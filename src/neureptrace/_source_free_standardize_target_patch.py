"""Normalize source-free adaptation config values consistently."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_free_standardize_target_patch_installed"
_INTEGER_ARRAY_PATCH_MARKER = "_neureptrace_source_free_integer_rejects_array_scalars"
_POSITIVE_FLOAT_ARRAY_PATCH_MARKER = "_neureptrace_source_free_positive_float_rejects_array_scalars"
_BOUNDED_FLOAT_ARRAY_PATCH_MARKER = "_neureptrace_source_free_bounded_float_rejects_array_scalars"


def _normalize_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        integer = int(value)
        if integer in {0, 1}:
            return bool(integer)
        raise ValueError(f"{name} must be a boolean value.")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean value.")


def _is_array_like_scalar(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, Mapping):
        return True
    if isinstance(value, (set, frozenset)):
        return True
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _reject_array_scalar(value: Any, *, name: str, expectation: str) -> None:
    if _is_array_like_scalar(value):
        raise ValueError(f"{name} must be {expectation}.")


def _install_numeric_array_guards(source_free: Any) -> None:
    original_integer = source_free._integer
    if not getattr(original_integer, _INTEGER_ARRAY_PATCH_MARKER, False):

        @wraps(original_integer)
        def _integer_checked(value: Any, name: str) -> int:
            _reject_array_scalar(value, name=name, expectation="an integer")
            return original_integer(value, name)

        setattr(_integer_checked, _INTEGER_ARRAY_PATCH_MARKER, True)
        source_free._integer = _integer_checked

    original_positive_float = source_free._positive_float
    if not getattr(original_positive_float, _POSITIVE_FLOAT_ARRAY_PATCH_MARKER, False):

        @wraps(original_positive_float)
        def _positive_float_checked(value: Any, name: str) -> float:
            _reject_array_scalar(value, name=name, expectation="positive and finite")
            return original_positive_float(value, name)

        setattr(_positive_float_checked, _POSITIVE_FLOAT_ARRAY_PATCH_MARKER, True)
        source_free._positive_float = _positive_float_checked

    original_bounded_float = source_free._bounded_float
    if not getattr(original_bounded_float, _BOUNDED_FLOAT_ARRAY_PATCH_MARKER, False):

        @wraps(original_bounded_float)
        def _bounded_float_checked(value: Any, name: str, *, lower: float, upper: float, include_upper: bool) -> float:
            if _is_array_like_scalar(value):
                closing = "]" if include_upper else ")"
                raise ValueError(f"{name} must be finite in [{lower}, {upper}{closing}.")
            return original_bounded_float(value, name, lower=lower, upper=upper, include_upper=include_upper)

        setattr(_bounded_float_checked, _BOUNDED_FLOAT_ARRAY_PATCH_MARKER, True)
        source_free._bounded_float = _bounded_float_checked


def _persist_normalized_numeric_config(source_free: Any, adapter: Any) -> None:
    """Keep accepted numeric config aliases usable after ``fit``.

    ``SourceFreeSubjectAdapter.fit`` already validates these values through the
    module-level normalizers, but it stores only local normalized variables.
    ``metadata`` and later predictions read from ``self`` again, so accepted
    integer-like strings such as ``"1.0"`` must be written back to the adapter.
    """

    adapter.confidence_threshold = source_free._bounded_float(
        adapter.confidence_threshold,
        "source_free_confidence_threshold",
        lower=0.0,
        upper=1.0,
        include_upper=True,
    )
    adapter.max_iterations = source_free._nonnegative_int(adapter.max_iterations, "source_free_max_iterations")
    adapter.min_class_count = source_free._positive_int(adapter.min_class_count, "source_free_min_class_count")
    adapter.min_active_classes = source_free._positive_int(adapter.min_active_classes, "source_free_min_active_classes")
    adapter.prototype_weight = source_free._bounded_float(
        adapter.prototype_weight,
        "source_free_prototype_weight",
        lower=0.0,
        upper=1.0,
        include_upper=True,
    )
    adapter.prototype_temperature = source_free._positive_float(adapter.prototype_temperature, "source_free_prototype_temperature")


def install() -> None:
    """Patch source-free adaptation config parsing."""

    source_free = importlib.import_module("neureptrace.decoding.source_free")
    _install_numeric_array_guards(source_free)
    adapter_cls = source_free.SourceFreeSubjectAdapter
    original_fit = adapter_cls.fit
    if getattr(original_fit, _PATCH_MARKER, False):
        return

    @wraps(original_fit)
    def fit(self, target_features, *, source_model=None, classes=None):
        self.standardize_target = _normalize_bool(self.standardize_target, name="source_free_standardize_target")
        result = original_fit(self, target_features, source_model=source_model, classes=classes)
        _persist_normalized_numeric_config(source_free, self)
        return result

    setattr(fit, _PATCH_MARKER, True)
    adapter_cls.fit = fit


__all__ = ["install"]
