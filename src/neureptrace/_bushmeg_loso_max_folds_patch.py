"""Reject invalid BUSH-MEG LOSO fold caps before slicing folds."""

from __future__ import annotations

import importlib
import math
from functools import wraps
from typing import Any

_PATCH_MARKER = "_neureptrace_bushmeg_loso_max_folds_patch_installed"


def _non_negative_int_or_none(value: Any, *, name: str = "max_folds") -> int | None:
    """Normalize optional fold caps without accepting negative slice semantics."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer or None.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer or None.") from exc
    if not math.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 0.0:
        raise ValueError(f"{name} must be a non-negative integer or None.")
    return int(parsed)


def install() -> None:
    """Install validation for the direct BUSH-MEG source-only LOSO runner."""

    module = importlib.import_module("neureptrace.bushmeg_loso_decode")
    if getattr(module, _PATCH_MARKER, False):
        return

    original_run = module.run_bushmeg_loso_decode

    @wraps(original_run)
    def run_bushmeg_loso_decode(*args, **kwargs):
        if "max_folds" in kwargs:
            kwargs = dict(kwargs)
            kwargs["max_folds"] = _non_negative_int_or_none(kwargs["max_folds"])
        return original_run(*args, **kwargs)

    module._non_negative_max_folds = _non_negative_int_or_none
    module.run_bushmeg_loso_decode = run_bushmeg_loso_decode
    setattr(module, _PATCH_MARKER, True)


__all__ = ["install"]
