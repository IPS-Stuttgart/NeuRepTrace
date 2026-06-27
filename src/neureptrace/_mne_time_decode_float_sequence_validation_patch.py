"""Validate MNE time-decode floating-point time sequences."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_mne_time_decode_float_sequence_validation_patch_installed"
_DEFAULT_NAME = "decode_candidate_times"


def _is_bool_scalar(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _validation_error(name: str) -> ValueError:
    return ValueError(f"{name} must contain finite numeric time values, not booleans or NaN/inf.")


def _coerce_sequence(value: Any, default: Sequence[float]) -> list[Any]:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return list(default)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return [part.strip() for chunk in text.split(",") for part in chunk.split() if part.strip()]
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return list(value)
    return [value]


def _parse_validated_float_sequence(value: Any, default: Sequence[float], *, name: str) -> tuple[float, ...]:
    values = _coerce_sequence(value, default)
    if any(_is_bool_scalar(item) for item in values):
        raise _validation_error(name)
    try:
        parsed = tuple(float(item) for item in values)
    except (TypeError, ValueError) as exc:
        raise _validation_error(name) from exc
    if not parsed:
        raise ValueError("Expected at least one time value.")
    if not all(np.isfinite(item) for item in parsed):
        raise _validation_error(name)
    return parsed


def install() -> None:
    """Patch candidate-time parsing to reject non-finite and boolean values."""

    module = importlib.import_module("neureptrace.mne_time_decode")
    original_parse_float_sequence = module._parse_float_sequence
    if getattr(original_parse_float_sequence, _PATCH_MARKER, False):
        return

    @wraps(original_parse_float_sequence)
    def _parse_float_sequence(
        value: object | Sequence[object] | None,
        *,
        default: Sequence[float],
        name: str = _DEFAULT_NAME,
    ) -> tuple[float, ...]:
        normalized_name = str(name).strip() or _DEFAULT_NAME
        return _parse_validated_float_sequence(value, default, name=normalized_name)

    setattr(_parse_float_sequence, _PATCH_MARKER, True)
    module._parse_float_sequence = _parse_float_sequence


__all__ = ["install"]
