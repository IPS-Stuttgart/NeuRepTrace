"""Normalize invalid reaction-time trial value errors."""

from __future__ import annotations

import importlib

import numpy as np

_PATCH_MARKER = "_neureptrace_reaction_time_trial_value_type_patch_installed"


def install() -> None:
    """Ensure invalid trial objects raise the documented ValueError."""

    module = importlib.import_module("neureptrace.behavior.reaction_time")
    original_to_int = module._to_int
    if getattr(original_to_int, _PATCH_MARKER, False):
        return

    def _to_int(value: object) -> int:
        text = "" if value is None else str(value).strip()
        try:
            number = float(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"trial values must be finite integers, got {value!r}.") from exc
        if not np.isfinite(number) or not number.is_integer():
            raise ValueError(f"trial values must be finite integers, got {value!r}.")
        return int(number)

    setattr(_to_int, _PATCH_MARKER, True)
    module._to_int = _to_int


__all__ = ["install"]
