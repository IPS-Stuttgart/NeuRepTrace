"""Normalize source-free adaptation config values consistently."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_free_standardize_target_patch_installed"


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
