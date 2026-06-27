"""Normalize conditional-CORAL boolean config values from CLI/YAML-style inputs."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_conditional_coral_bool_config_patch_installed"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}


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


def install() -> None:
    """Install strict boolean normalization for conditional-CORAL config."""

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
        random_state: int | str | None = 13,
    ):
        return original_config(
            regularization=regularization,
            min_target_rows_per_class=min_target_rows_per_class,
            confidence_threshold=confidence_threshold,
            fallback=fallback,
            center=_normalize_bool(center, name="center"),
            random_state=random_state,
        )

    setattr(conditional_coral_config, _PATCH_MARKER, True)
    conditional_coral.conditional_coral_config = conditional_coral_config


__all__ = ["install"]
