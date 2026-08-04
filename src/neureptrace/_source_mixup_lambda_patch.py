"""Validate Source MixUp lambda weights and preserve disabled feature precision."""

from __future__ import annotations

import importlib
from dataclasses import replace
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_mixup_lambda_boolean_patch_installed"
_OUTPUT_PATCH_MARKER = "_neureptrace_source_mixup_disabled_output_patch_installed"
_ERROR = "lambdas must be finite numeric values in [0, 1], not booleans."


def _materialize(value: Any) -> Any:
    if value is None or isinstance(value, (str, bytes, np.ndarray)):
        return value
    try:
        return tuple(value)
    except TypeError:
        return value


def _materialize_features(value: Any) -> Any:
    """Expand generator-backed feature rows exactly once."""

    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (str, bytes)):
        return value
    try:
        iterator = iter(value)
    except TypeError:
        return value
    return [_materialize_features(item) for item in iterator]


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


def _compact_float32(values: np.ndarray) -> np.ndarray:
    """Use float32 only when every finite nonzero feature survives."""

    array = np.asarray(values)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        compact = array.astype(np.float32, copy=False)
    if np.any(np.isfinite(array) & ~np.isfinite(compact)):
        return array
    if np.any((array != 0.0) & (compact == 0.0)):
        return array
    return compact


def install() -> None:
    """Install Source MixUp lambda guards and lossless disabled output handling."""

    source_mixup = importlib.import_module("neureptrace.decoding.source_mixup")
    original_mixup_rows = source_mixup.mixup_rows
    if not getattr(original_mixup_rows, _PATCH_MARKER, False):

        @wraps(original_mixup_rows)
        def mixup_rows(content_features, partner_features, *, lambdas):
            lambda_values = _materialize(lambdas)
            if _contains_boolean(lambda_values):
                raise ValueError(_ERROR)
            return original_mixup_rows(content_features, partner_features, lambdas=lambda_values)

        setattr(mixup_rows, _PATCH_MARKER, True)
        source_mixup.mixup_rows = mixup_rows

    original_augment_source_with_mixup = source_mixup.augment_source_with_mixup
    if not getattr(original_augment_source_with_mixup, _OUTPUT_PATCH_MARKER, False):

        @wraps(original_augment_source_with_mixup)
        def augment_source_with_mixup(
            source_features,
            source_labels,
            *,
            source_domains=None,
            config=None,
        ):
            materialized_features = _materialize_features(source_features)
            result = original_augment_source_with_mixup(
                materialized_features,
                source_labels,
                source_domains=source_domains,
                config=config,
            )
            if result.metadata["source_mixup"]:
                return result
            features = source_mixup._feature_matrix(materialized_features, name="source_features")
            return replace(result, features=_compact_float32(features))

        setattr(augment_source_with_mixup, _OUTPUT_PATCH_MARKER, True)
        source_mixup.augment_source_with_mixup = augment_source_with_mixup

    # Package initialization calls this installer immediately after the source-roll
    # compatibility patch, so the final public wrapper can be corrected here.
    source_roll_patch = importlib.import_module("neureptrace._source_roll_disabled_output_patch")
    source_roll_patch.install()


__all__ = ["install"]
