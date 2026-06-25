"""Accept numeric boolean config values in the legacy workflow parser."""

from __future__ import annotations

import importlib
import math
from functools import wraps
from numbers import Real
from typing import Any

_PATCH_MARKER = "_neureptrace_config_workflow_numeric_bool_patch_installed"
_TRUE_STRINGS = {"1", "true", "yes", "on"}
_FALSE_STRINGS = {"0", "false", "no", "off"}


def install() -> None:
    """Install tolerant boolean parsing for legacy dataset workflow configs."""

    config_workflow = importlib.import_module("neureptrace.config_workflow")
    original = config_workflow._as_bool
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def _as_bool(value: Any, *, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, Real):
            numeric = float(value)
            if math.isfinite(numeric) and numeric in {0.0, 1.0}:
                return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in _TRUE_STRINGS:
                return True
            if normalized in _FALSE_STRINGS:
                return False
        raise config_workflow.DatasetConfigError(f"Cannot interpret {value!r} as a boolean.")

    setattr(_as_bool, _PATCH_MARKER, True)
    config_workflow._as_bool = _as_bool


__all__ = ["install"]
