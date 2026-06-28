"""Accept tolerant legacy workflow scalar and window config values."""

from __future__ import annotations

import importlib
import math
from functools import wraps
from numbers import Real
from typing import Any

_BOOL_PATCH_MARKER = "_neureptrace_config_workflow_float_bool_patch_installed"
_FLOAT_PAIR_PATCH_MARKER = "_neureptrace_config_workflow_string_pair_patch_installed"


def _install_domain_importance_bool_config_patch() -> None:
    domain_patch = importlib.import_module("neureptrace._domain_importance_bool_config_patch")
    domain_patch.install()


def _string_pair_values(value: str) -> list[str] | None:
    text = value.strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [part.strip() for comma_part in text.split(",") for part in comma_part.split() if part.strip()]


def install() -> None:
    """Install tolerant parsing for generated dataset workflow configs."""

    _install_domain_importance_bool_config_patch()
    config_workflow = importlib.import_module("neureptrace.config_workflow")

    original_bool = config_workflow._as_bool
    if not getattr(original_bool, _BOOL_PATCH_MARKER, False):

        @wraps(original_bool)
        def _as_bool(value: Any, *, default: bool = False) -> bool:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, Real):
                numeric = float(value)
                if math.isfinite(numeric) and numeric in {0.0, 1.0}:
                    return bool(value)
            return original_bool(value, default=default)

        setattr(_as_bool, _BOOL_PATCH_MARKER, True)
        config_workflow._as_bool = _as_bool

    original_float_pair = config_workflow._as_float_pair
    if not getattr(original_float_pair, _FLOAT_PAIR_PATCH_MARKER, False):

        @wraps(original_float_pair)
        def _as_float_pair(value: Any, *, name: str) -> tuple[float, float] | None:
            if isinstance(value, str):
                parsed = _string_pair_values(value)
                if parsed is None:
                    return None
                return original_float_pair(parsed, name=name)
            return original_float_pair(value, name=name)

        setattr(_as_float_pair, _FLOAT_PAIR_PATCH_MARKER, True)
        config_workflow._as_float_pair = _as_float_pair


__all__ = ["install"]
