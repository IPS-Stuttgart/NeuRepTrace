"""Normalize optional random-state values for generative augmentation configs."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_INSTALLED = False
_PATCH_MARKER = "_neureptrace_generative_augmentation_random_state_patch_installed"


def _random_state_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a non-negative integer or None.")


def _is_none_random_state(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _scalar_random_state_value(value: Any, *, name: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _random_state_error(name)
        return value.item()
    if isinstance(value, (list, tuple, dict, set)):
        raise _random_state_error(name)
    return value


def _normalize_optional_nonnegative_int(value: Any, *, name: str = "random_state") -> int | None:
    value = _scalar_random_state_value(value, name=name)
    if _is_none_random_state(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise _random_state_error(name)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise _random_state_error(name) from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0 or numeric < 0.0:
        raise _random_state_error(name)
    return int(numeric)


def install() -> None:
    """Patch generative augmentation config random-state normalization."""

    global _INSTALLED
    if _INSTALLED:
        return

    module = importlib.import_module("neureptrace.decoding.generative_augmentation")
    original_config = module.generative_augmentation_config
    if getattr(original_config, _PATCH_MARKER, False):
        _INSTALLED = True
        return

    @wraps(original_config)
    def generative_augmentation_config(*args: Any, **kwargs: Any):
        if "random_state" in kwargs:
            kwargs = dict(kwargs)
            kwargs["random_state"] = _normalize_optional_nonnegative_int(kwargs["random_state"], name="random_state")
        return original_config(*args, **kwargs)

    setattr(generative_augmentation_config, _PATCH_MARKER, True)
    module.generative_augmentation_config = generative_augmentation_config
    _INSTALLED = True


__all__ = ["install"]
