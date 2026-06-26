"""Normalize invalid reaction-time trial value errors."""

from __future__ import annotations

import importlib

import numpy as np

_PATCH_MARKER = "_neureptrace_reaction_time_trial_value_type_patch_installed"


def _trial_error(value: object) -> ValueError:
    return ValueError(f"trial values must be finite integers, got {value!r}.")


def install() -> None:
    """Ensure invalid trial objects raise the documented ValueError."""

    module = importlib.import_module("neureptrace.behavior.reaction_time")
    original_to_int = module._to_int
    if getattr(original_to_int, _PATCH_MARKER, False):
        return

    def _to_int(value: object) -> int:
        try:
            text = "" if value is None else str(value).strip()
            number = float(text)
        except (TypeError, ValueError) as exc:
            raise _trial_error(value) from exc
        if not np.isfinite(number) or not number.is_integer():
            raise _trial_error(value)
        return int(number)

    setattr(_to_int, _PATCH_MARKER, True)
    module._to_int = _to_int


__all__ = ["install"]
