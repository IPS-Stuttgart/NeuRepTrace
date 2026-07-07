"""Reject boolean source-free probability inputs."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_free_probability_bool_patch_installed"


def _materialize_nested_iterables(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        materialized = [_materialize_nested_iterables(item) for item in value.ravel(order="C")]
        return np.asarray(materialized, dtype=object).reshape(value.shape)
    if isinstance(value, (str, bytes)):
        return value
    if not isinstance(value, Iterable):
        return value
    return [_materialize_nested_iterables(item) for item in value]


def _contains_boolean_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return True
        if value.dtype == object:
            return any(_contains_boolean_value(item) for item in value.flat)
        return False
    if isinstance(value, (str, bytes)):
        return False
    if not isinstance(value, Iterable):
        return False
    return any(_contains_boolean_value(item) for item in value)


def _reject_boolean_probabilities(value: Any, *, source: str) -> None:
    materialized = _materialize_nested_iterables(value)
    if _contains_boolean_value(materialized):
        raise ValueError(f"{source} must contain numeric probability values, not boolean flags.")


def install() -> None:
    source_free = importlib.import_module("neureptrace.decoding.source_free")
    if getattr(source_free, _PATCH_MARKER, False):
        return

    original_predict_source_probabilities = source_free._predict_source_probabilities
    original_normalize_probability_rows = source_free._normalize_probability_rows

    @wraps(original_predict_source_probabilities)
    def _predict_source_probabilities(model: Any, features: np.ndarray, classes: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            _reject_boolean_probabilities(model.predict_proba(features), source="source_model probabilities")
        return original_predict_source_probabilities(model, features, classes)

    @wraps(original_normalize_probability_rows)
    def _normalize_probability_rows(probabilities: Any) -> np.ndarray:
        _reject_boolean_probabilities(probabilities, source="probabilities")
        return original_normalize_probability_rows(probabilities)

    source_free._predict_source_probabilities = _predict_source_probabilities
    source_free._normalize_probability_rows = _normalize_probability_rows
    setattr(source_free, _PATCH_MARKER, True)
