"""Reject boolean Source MixUp lambda weights before numeric coercion."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_mixup_lambda_boolean_patch_installed"
_ERROR = "lambdas must be finite numeric values in [0, 1], not booleans."


def _materialize(value: Any) -> Any:
    if value is None or isinstance(value, (str, bytes, np.ndarray)):
        return value
    try:
        return tuple(value)
    except TypeError:
        return value


def _contains_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return True
        if value.dtype == object:
            return any(_contains_boolean(item) for item in value.ravel(order="C"))
        return False
    if value is None or isinstance(value, (str, bytes)):
        return False
    try:
        return any(_contains_boolean(item) for item in value)
    except TypeError:
        return False


def install() -> None:
    """Install a guard for public MixUp lambda weights."""

    source_mixup = importlib.import_module("neureptrace.decoding.source_mixup")
    original_mixup_rows = source_mixup.mixup_rows
    if getattr(original_mixup_rows, _PATCH_MARKER, False):
        return

    @wraps(original_mixup_rows)
    def mixup_rows(content_features, partner_features, *, lambdas):
        lambda_values = _materialize(lambdas)
        if _contains_boolean(lambda_values):
            raise ValueError(_ERROR)
        return original_mixup_rows(content_features, partner_features, lambdas=lambda_values)

    setattr(mixup_rows, _PATCH_MARKER, True)
    source_mixup.mixup_rows = mixup_rows


__all__ = ["install"]
