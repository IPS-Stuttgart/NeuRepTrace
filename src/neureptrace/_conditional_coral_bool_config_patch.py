"""Normalize conditional-CORAL config values from CLI/YAML-style inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_conditional_coral_bool_config_patch_installed"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}
_NONE_STRINGS = {"", "none", "null"}


def _bool_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a boolean value.")


def _normalize_bool(value: Any, *, name: str) -> bool:
    """Return a real bool while rejecting ambiguous truthy/falsy objects."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
        raise _bool_error(name)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _bool_error(name)
        return _normalize_bool(value.item(), name=name)
    if isinstance(value, (int, np.integer)):
        if int(value) in {0, 1}:
            return bool(value)
        raise _bool_error(name)
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value) in {0.0, 1.0}:
            return bool(value)
        raise _bool_error(name)
    raise _bool_error(name)


def _random_state_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a non-negative integer or None.")


def _normalize_optional_random_state(value: Any, *, name: str) -> int | None:
    """Normalize optional integer seeds without leaking raw set/NumPy errors."""

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in _NONE_STRINGS:
            return None
        value = stripped
    elif isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _random_state_error(name)
        return _normalize_optional_random_state(value.item(), name=name)
    elif isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
        raise _random_state_error(name)
    if isinstance(value, (bool, np.bool_)):
        raise _random_state_error(name)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise _random_state_error(name) from exc
    if not np.isfinite(parsed) or parsed < 0.0 or parsed % 1.0 != 0.0:
        raise _random_state_error(name)
    return int(parsed)


def install() -> None:
    """Install strict normalization for conditional-CORAL config."""

    from neureptrace.decoding import conditional_coral

    original_config = conditional_coral.conditional_coral_config
    if getattr(original_config, _PATCH_MARKER, False):
        return

    @wraps(original_config)
    def conditional_coral_config(
        *,
        regularization: float | str = conditional_coral.DEFAULT_CONDITIONAL_CORAL_REGULARIZATION,
        min_target_rows_per_class: int | str = conditional_coral.DEFAULT_CONDITIONAL_CORAL_MIN_TARGET_ROWS,
        confidence_threshold: float | str = 0.0,
        fallback: str = "global",
        center: Any = True,
        random_state: Any = 13,
    ):
        return original_config(
            regularization=regularization,
            min_target_rows_per_class=min_target_rows_per_class,
            confidence_threshold=confidence_threshold,
            fallback=fallback,
            center=_normalize_bool(center, name="center"),
            random_state=_normalize_optional_random_state(random_state, name="random_state"),
        )

    setattr(conditional_coral_config, _PATCH_MARKER, True)
    conditional_coral.conditional_coral_config = conditional_coral_config


__all__ = ["install"]
