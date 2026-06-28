"""Normalize OpenNeuro real-vs-shuffle boolean provenance tokens."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

_PATCH_MARKER = "_neureptrace_openneuro_real_shuffle_bool_patch_installed"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}
_MISSING_STRINGS = {"", "none", "null", "nan", "na", "n/a"}


def _is_missing(value: Any) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _as_bool_token(value: object) -> bool:
    """Parse CSV/JSON provenance booleans without silently accepting junk tokens."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or _is_missing(value):
        return False
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"Cannot parse non-scalar boolean provenance value {value!r}.")
        return _as_bool_token(value.item())
    if isinstance(value, (int, np.integer)):
        parsed = int(value)
        if parsed in {0, 1}:
            return bool(parsed)
        raise ValueError(f"Cannot parse boolean provenance value {value!r}.")
    if isinstance(value, (float, np.floating)):
        parsed = float(value)
        if np.isfinite(parsed) and parsed in {0.0, 1.0}:
            return bool(parsed)
        raise ValueError(f"Cannot parse boolean provenance value {value!r}.")

    text = str(value).strip().lower()
    if text in _MISSING_STRINGS:
        return False
    if text in _TRUE_STRINGS:
        return True
    if text in _FALSE_STRINGS:
        return False
    try:
        numeric = float(text)
    except ValueError:
        numeric = None
    if numeric is not None and np.isfinite(numeric) and numeric in {0.0, 1.0}:
        return bool(numeric)
    raise ValueError(f"Cannot parse boolean provenance value {value!r}.")


def install() -> None:
    """Install robust boolean parsing for real-vs-shuffle provenance checks."""

    report = importlib.import_module("neureptrace.openneuro_real_shuffle_report")
    if getattr(report, _PATCH_MARKER, False):
        return
    report._as_bool_token = _as_bool_token
    setattr(report, _PATCH_MARKER, True)


__all__ = ["install"]
