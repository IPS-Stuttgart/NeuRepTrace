"""Compatibility for numeric boolean values in config workflow files."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

_MARKER = "_neureptrace_config_workflow_numeric_bool_installed"


def install() -> None:
    module = importlib.import_module("neureptrace.config_workflow")
    original = module._as_bool
    if getattr(original, _MARKER, False):
        return

    @wraps(original)
    def as_bool(value: Any, *, default: bool = False) -> bool:
        if value in (0, 1) and not isinstance(value, bool):
            return bool(value)
        return original(value, default=default)

    setattr(as_bool, _MARKER, True)
    setattr(module, "_as_bool", as_bool)


__all__ = ["install"]
