"""Validate source-free source-model probability matrices."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_free_probability_rows_patch_installed"
_FUNC_MARKER = "_neureptrace_source_free_probability_rows_patch_wrapped"
_NORMALIZE_MARKER = "_neureptrace_source_free_probability_rows_normalize_wrapped"
_NEGATIVE_TOLERANCE = 1e-10


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
    importlib.import_module("neureptrace._source_quantile_bin_dtype_patch").install()


__all__ = ["install"]
