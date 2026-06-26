"""Robust validation for optional source-selection bound controls."""

from __future__ import annotations

from typing import Any

import numpy as np

_PATCH_ATTR = "_neureptrace_source_selection_optional_bounds_patch"


def _scalar_config_value(value: Any, *, name: str, message: str) -> Any:
    """Return a scalar config value or raise a user-facing ValueError."""

    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        return value.item()
    if isinstance(value, (list, tuple, dict, set)):
        raise ValueError(message)
    return value


def _is_optional_sentinel(value: Any, aliases: set[str]) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in aliases
    return False


def install() -> None:
    """Install scalar-safe optional bound normalization for source selection."""

    from neureptrace.decoding import source_selection

    if getattr(source_selection._normalize_optional_positive_int, _PATCH_ATTR, False):
        return

    original_optional_positive = source_selection._normalize_optional_positive_int
    original_optional_float = source_selection._normalize_optional_nonnegative_float
    normalize_positive_int = source_selection._normalize_positive_int

    def _normalize_optional_positive_int(value: int | str | None, *, name: str) -> int | None:
        message = f"{name} must be a positive integer."
        scalar = _scalar_config_value(value, name=name, message=message)
        if _is_optional_sentinel(scalar, {"", "none", "null", "all", "full"}):
            return None
        return normalize_positive_int(scalar, name=name)

    def _normalize_optional_nonnegative_float(value: float | str | None, *, name: str) -> float | None:
        message = f"{name} must be finite and non-negative."
        scalar = _scalar_config_value(value, name=name, message=message)
        if _is_optional_sentinel(scalar, {"", "none", "null", "off"}):
            return None
        if isinstance(scalar, (bool, np.bool_)):
            raise ValueError(message)
        try:
            parsed = float(scalar)
        except (TypeError, ValueError) as exc:
            raise ValueError(message) from exc
        if not np.isfinite(parsed) or parsed < 0.0:
            raise ValueError(message)
        return parsed

    setattr(_normalize_optional_positive_int, _PATCH_ATTR, True)
    setattr(_normalize_optional_nonnegative_float, _PATCH_ATTR, True)
    _normalize_optional_positive_int.__wrapped__ = original_optional_positive
    _normalize_optional_nonnegative_float.__wrapped__ = original_optional_float
    source_selection._normalize_optional_positive_int = _normalize_optional_positive_int
    source_selection._normalize_optional_nonnegative_float = _normalize_optional_nonnegative_float


__all__ = ["install"]
