"""Allow structured OpenNeuro alignment manifest fallback values."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import pandas as pd

_PATCH_MARKER = "_neureptrace_openneuro_alignment_compare_structured_values_patch_installed"

_TRUE_TOKENS = {"1", "true", "yes", "y", "on"}
_FALSE_TOKENS = {"0", "false", "no", "n", "off"}


def _is_missing_scalar(value: Any) -> bool:
    """Return true for scalar pandas/numpy missing values only."""

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, bool) else False


def _first_nonempty(*values: Any) -> str:
    """Return the first non-empty value as text without hashing structured values."""

    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _bool_tokens(value: Any) -> list[str]:
    """Normalize scalar or sequence boolean-like values into lowercase tokens."""

    if value is None or _is_missing_scalar(value):
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(token).strip().lower() for token in value if str(token).strip()]
    return [str(value).strip().lower()] if str(value).strip() else []


def _as_bool(value: Any) -> bool:
    """Parse scalar or sequence manifest/CSV boolean values without ambiguous pd.isna checks."""

    if isinstance(value, bool):
        return value
    tokens = _bool_tokens(value)
    if not tokens:
        return False
    parsed: set[bool] = set()
    for token in tokens:
        if token in _TRUE_TOKENS:
            parsed.add(True)
        elif token in _FALSE_TOKENS:
            parsed.add(False)
        else:
            parsed.add(False)
    if len(parsed) > 1:
        raise ValueError(f"Inconsistent boolean provenance: {tokens}")
    return parsed.pop()


def install() -> None:
    """Patch OpenNeuro alignment comparison structured-value handling."""

    module = importlib.import_module("neureptrace.openneuro_alignment_compare")
    if getattr(module, _PATCH_MARKER, False):
        return
    module._first_nonempty = _first_nonempty
    module._as_bool = _as_bool
    setattr(module, _PATCH_MARKER, True)


__all__ = ["install"]
