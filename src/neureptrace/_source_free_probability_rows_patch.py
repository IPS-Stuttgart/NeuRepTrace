"""Validate source-free probability matrices and iterable inputs."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_free_probability_rows_patch_installed"
_FUNC_MARKER = "_neureptrace_source_free_probability_rows_patch_wrapped"
_NORMALIZE_MARKER = "_neureptrace_source_free_probability_rows_normalize_wrapped"
_CONSENSUS_PATCH_MARKER = "_neureptrace_source_free_consensus_one_pass_patch_installed"
_CONSENSUS_TENSOR_MARKER = "_neureptrace_source_free_consensus_probability_tensor_wrapped"
_CONSENSUS_NORMALIZE_MARKER = "_neureptrace_source_free_consensus_normalize_rows_wrapped"
_CONSENSUS_WEIGHTS_MARKER = "_neureptrace_source_free_consensus_weights_wrapped"
_NEGATIVE_TOLERANCE = 1e-10


def _materialize_one_pass_iterables(value: Any) -> Any:
    """Return a re-iterable representation for generator-style numeric inputs."""

    if isinstance(value, (np.ndarray, str, bytes, Mapping)):
        return value
    try:
        iterator = iter(value)
    except TypeError:
        return value
    return tuple(_materialize_one_pass_iterables(item) for item in iterator)


def _install_source_free_consensus_probability_iterables() -> None:
    """Patch source-free consensus probability helpers to accept one-pass iterables."""

    source_free_consensus = importlib.import_module("neureptrace.decoding.source_free_consensus")

    original_normalize_probability_rows = source_free_consensus._normalize_probability_rows
    if not getattr(original_normalize_probability_rows, _CONSENSUS_NORMALIZE_MARKER, False):

        @wraps(original_normalize_probability_rows)
        def _normalize_probability_rows(probabilities: Any) -> np.ndarray:
            return original_normalize_probability_rows(_materialize_one_pass_iterables(probabilities))

        setattr(_normalize_probability_rows, _CONSENSUS_NORMALIZE_MARKER, True)
        source_free_consensus._normalize_probability_rows = _normalize_probability_rows

    original_normalize_weights = source_free_consensus._normalize_weights
    if not getattr(original_normalize_weights, _CONSENSUS_WEIGHTS_MARKER, False):

        @wraps(original_normalize_weights)
        def _normalize_weights(weights: Any, *, n_variants: int) -> np.ndarray:
            return original_normalize_weights(_materialize_one_pass_iterables(weights), n_variants=n_variants)

        setattr(_normalize_weights, _CONSENSUS_WEIGHTS_MARKER, True)
        source_free_consensus._normalize_weights = _normalize_weights

    original_probability_tensor = source_free_consensus._probability_tensor
    if not getattr(original_probability_tensor, _CONSENSUS_TENSOR_MARKER, False):

        @wraps(original_probability_tensor)
        def _probability_tensor(probability_variants: Any) -> np.ndarray:
            try:
                variant_items = tuple(_materialize_one_pass_iterables(matrix) for matrix in probability_variants)
            except TypeError as exc:
                raise ValueError("At least one probability matrix is required.") from exc
            return original_probability_tensor(variant_items)

        setattr(_probability_tensor, _CONSENSUS_TENSOR_MARKER, True)
        source_free_consensus._probability_tensor = _probability_tensor

    setattr(source_free_consensus, _CONSENSUS_PATCH_MARKER, True)


def install() -> None:
    """Patch source-free probability prediction validation."""

    source_free = importlib.import_module("neureptrace.decoding.source_free")

    original_normalize_probability_rows = source_free._normalize_probability_rows
    if not getattr(original_normalize_probability_rows, _NORMALIZE_MARKER, False):

        @wraps(original_normalize_probability_rows)
        def _normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
            matrix = np.asarray(probabilities, dtype=float)
            if np.all(np.isfinite(matrix)) and np.any(matrix < -_NEGATIVE_TOLERANCE):
                raise ValueError("probabilities must be non-negative.")
            if np.all(np.isfinite(matrix)) and np.any(matrix < 0.0):
                matrix = np.where(matrix < 0.0, 0.0, matrix)
            return original_normalize_probability_rows(matrix)

        setattr(_normalize_probability_rows, _NORMALIZE_MARKER, True)
        source_free._normalize_probability_rows = _normalize_probability_rows

    original_predict_source_probabilities = source_free._predict_source_probabilities
    if getattr(original_predict_source_probabilities, _FUNC_MARKER, False):
        setattr(source_free, _PATCH_MARKER, True)
        _install_source_free_consensus_probability_iterables()
        importlib.import_module("neureptrace._source_quantile_bin_dtype_patch").install()
        return

    @wraps(original_predict_source_probabilities)
    def _predict_source_probabilities(model: Any, features: np.ndarray, classes: np.ndarray) -> np.ndarray:
        probabilities = original_predict_source_probabilities(model, features, classes)
        expected_rows = np.asarray(features).shape[0]
        if probabilities.shape[0] != expected_rows:
            raise ValueError("source_model probability rows must match target_features rows.")
        return probabilities

    setattr(_predict_source_probabilities, _FUNC_MARKER, True)
    source_free._predict_source_probabilities = _predict_source_probabilities
    setattr(source_free, _PATCH_MARKER, True)
    _install_source_free_consensus_probability_iterables()
    importlib.import_module("neureptrace._source_quantile_bin_dtype_patch").install()


__all__ = ["install"]
