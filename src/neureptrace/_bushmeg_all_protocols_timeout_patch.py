"""Validate BUSH-MEG all-protocol timeout controls strictly."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_bushmeg_all_protocols_timeout_patch_installed"


def _is_bool_scalar(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _validate_timeout_seconds(name: str, value: float | int | None) -> float | None:
    if value is None:
        return None
    if _is_bool_scalar(value):
        raise ValueError(f"{name} must be a positive finite number when provided.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number when provided.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive finite number when provided.")
    return parsed


def install() -> None:
    """Patch BUSH-MEG all-protocol timeout validation."""

    all_protocols = importlib.import_module("neureptrace.bushmeg_all_protocols")
    if getattr(all_protocols, _PATCH_MARKER, False):
        return

    all_protocols._validate_timeout_seconds = _validate_timeout_seconds
    setattr(all_protocols, _PATCH_MARKER, True)


__all__ = ["install"]
