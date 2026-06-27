"""Accept float zero/one boolean config values in the legacy workflow parser."""

from __future__ import annotations

import importlib
import math
from functools import wraps
from numbers import Real
from typing import Any

_PATCH_MARKER = "_neureptrace_config_workflow_float_bool_patch_installed"


def _install_domain_importance_bool_config_patch() -> None:
    domain_patch = importlib.import_module("neureptrace._domain_importance_bool_config_patch")
    domain_patch.install()


def install() -> None:
    """Install tolerant boolean parsing for generated dataset workflow configs."""

    _install_domain_importance_bool_config_patch()
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
        return original(value, default=default)

    setattr(_as_bool, _PATCH_MARKER, True)
    config_workflow._as_bool = _as_bool


__all__ = ["install"]
