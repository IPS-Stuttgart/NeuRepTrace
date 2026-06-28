"""Normalize optional confidence-selection integer sentinels."""

from __future__ import annotations

from typing import Any

import numpy as np

from neureptrace.decoding import confidence_selection as _confidence_selection

_OPTIONAL_INT_SENTINELS = {"", "none", "null", "all", "full"}


def _optional_positive_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in _OPTIONAL_INT_SENTINELS:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    parsed = int(numeric)
    if parsed < 1:
        raise ValueError(f"{name} must be positive.")
    return parsed


def install() -> None:
    """Install the normalized optional integer parser."""

    _confidence_selection._optional_positive_int = _optional_positive_int
