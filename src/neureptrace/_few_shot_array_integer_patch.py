"""Runtime guardrails for array-valued few-shot integer controls.

NumPy arrays with one element are accepted by ``float(...)`` in current NumPy
releases.  That means malformed config values such as ``np.array([1])`` can be
silently coerced into scalar few-shot counts or seeds.  The core validators
already reject Python and NumPy boolean scalars; this patch closes the analogous
array-valued path while preserving ordinary NumPy numeric scalar values.
"""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_few_shot_array_integer_patch_installed"
_WRAPPER_MARKER = "_neureptrace_few_shot_array_integer_validator_wrapped"


def _reject_array_value(value: Any, *, name: str, kind: str) -> None:
    if isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a {kind} integer scalar, not an array.")


def _patch_validator(module: ModuleType, validator_name: str, *, kind: str) -> None:
    original = getattr(module, validator_name, None)
    if original is None or getattr(original, _WRAPPER_MARKER, False):
        return

    @wraps(original)
    def _wrapped(value: Any, *args: Any, **kwargs: Any):
        name = kwargs.get("name")
        if name is None and args:
            name = args[0]
        if name is None:
            name = "value"
        _reject_array_value(value, name=str(name), kind=kind)
        return original(value, *args, **kwargs)

    setattr(_wrapped, _WRAPPER_MARKER, True)
    setattr(module, validator_name, _wrapped)


def install() -> None:
    """Install array-valued few-shot integer-control validation."""

    from neureptrace.decoding import few_shot

    if getattr(few_shot, _PATCH_MARKER, False):
        return

    _patch_validator(few_shot, "_normalize_positive_int", kind="positive")
    _patch_validator(few_shot, "_normalize_nonnegative_int", kind="non-negative")
    setattr(few_shot, _PATCH_MARKER, True)


__all__ = ["install"]
