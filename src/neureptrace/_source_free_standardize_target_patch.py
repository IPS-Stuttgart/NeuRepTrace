"""Normalize source-free ``standardize_target`` values consistently."""

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


def install() -> None:
    """Patch source-free adaptation boolean option parsing."""

    source_free = importlib.import_module("neureptrace.decoding.source_free")
    adapter_cls = source_free.SourceFreeSubjectAdapter
    original_fit = adapter_cls.fit
    if getattr(original_fit, _PATCH_MARKER, False):
        return

    @wraps(original_fit)
    def fit(self, target_features, *, source_model=None, classes=None):
        self.standardize_target = _normalize_bool(self.standardize_target, name="source_free_standardize_target")
        return original_fit(self, target_features, source_model=source_model, classes=classes)

    setattr(fit, _PATCH_MARKER, True)
    adapter_cls.fit = fit


__all__ = ["install"]
