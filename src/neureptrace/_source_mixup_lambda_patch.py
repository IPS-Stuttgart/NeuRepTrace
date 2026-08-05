"""Reject invalid Source MixUp values before lossy numeric coercion."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from functools import wraps
from typing import Any

import numpy as np

_MIXUP_ROWS_PATCH_MARKER = "_neureptrace_source_mixup_lambda_validation_patch_installed"
_FEATURE_MATRIX_PATCH_MARKER = "_neureptrace_source_mixup_feature_validation_patch_installed"
_BOOLEAN_LAMBDA_ERROR = "lambdas must be finite numeric values in [0, 1], not booleans."
_COMPLEX_LAMBDA_ERROR = "lambdas must contain real-valued values, not complex values."


def _materialize(value: Any) -> Any:
    """Materialize one-pass nested inputs once so validation cannot consume them."""

    if value is None or isinstance(value, (str, bytes, np.ndarray, Mapping)):
        return value
    try:
        iterator = iter(value)
    except TypeError:
        return value
    return tuple(_materialize(item) for item in iterator)


def _contains_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return True
        if value.dtype == object:
            return any(_contains_boolean(item) for item in value.ravel(order="C"))
        return False
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return False
    try:
        return any(_contains_boolean(item) for item in value)
    except TypeError:
        return False


def _contains_complex(value: Any) -> bool:
    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return bool(value.size)
        if value.dtype == object:
            return any(_contains_complex(item) for item in value.ravel(order="C"))
        return False
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return False
    try:
        return any(_contains_complex(item) for item in value)
    except TypeError:
        return False


def install() -> None:
    """Install Source MixUp and Source Feature Roll input guards."""

    source_mixup = importlib.import_module("neureptrace.decoding.source_mixup")

    original_feature_matrix = source_mixup._feature_matrix
    if not getattr(original_feature_matrix, _FEATURE_MATRIX_PATCH_MARKER, False):

        @wraps(original_feature_matrix)
        def feature_matrix(values, *, name):
            materialized = _materialize(values)
            if _contains_complex(materialized):
                raise ValueError(f"{name} must contain real-valued feature values, not complex values.")
            return original_feature_matrix(materialized, name=name)

        setattr(feature_matrix, _FEATURE_MATRIX_PATCH_MARKER, True)
        source_mixup._feature_matrix = feature_matrix

    original_mixup_rows = source_mixup.mixup_rows
    if not getattr(original_mixup_rows, _MIXUP_ROWS_PATCH_MARKER, False):

        @wraps(original_mixup_rows)
        def mixup_rows(content_features, partner_features, *, lambdas):
            lambda_values = _materialize(lambdas)
            if _contains_boolean(lambda_values):
                raise ValueError(_BOOLEAN_LAMBDA_ERROR)
            if _contains_complex(lambda_values):
                raise ValueError(_COMPLEX_LAMBDA_ERROR)
            return original_mixup_rows(content_features, partner_features, lambdas=lambda_values)

        setattr(mixup_rows, _MIXUP_ROWS_PATCH_MARKER, True)
        source_mixup.mixup_rows = mixup_rows

    # Package initialization calls this installer immediately after the source-roll
    # compatibility patch, so the final public wrapper can be corrected here.
    source_roll_patch = importlib.import_module("neureptrace._source_roll_disabled_output_patch")
    source_roll_patch.install()
    source_roll_numeric_patch = importlib.import_module("neureptrace._source_roll_numeric_input_patch")
    source_roll_numeric_patch.install()


__all__ = ["install"]
