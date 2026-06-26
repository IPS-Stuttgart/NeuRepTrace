"""Normalize source-selection ``class_balance`` flag values."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_selection_class_balance_patch_installed"


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
    """Patch source-domain selection boolean option parsing."""

    source_selection = importlib.import_module("neureptrace.decoding.source_selection")
    original_select = source_selection.select_source_domains_by_target_similarity
    if getattr(original_select, _PATCH_MARKER, False):
        return

    @wraps(original_select)
    def select_source_domains_by_target_similarity(*args, **kwargs):
        class_balance = kwargs.pop("class_balance", False)
        kwargs["class_balance"] = _normalize_bool(class_balance, name="class_balance")
        return original_select(*args, **kwargs)

    setattr(select_source_domains_by_target_similarity, _PATCH_MARKER, True)
    source_selection.select_source_domains_by_target_similarity = select_source_domains_by_target_similarity


__all__ = ["install"]
