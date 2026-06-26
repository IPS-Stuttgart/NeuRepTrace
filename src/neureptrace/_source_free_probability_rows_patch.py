"""Validate source-free source-model probability row counts."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_free_probability_rows_patch_installed"
_FUNC_MARKER = "_neureptrace_source_free_probability_rows_patch_wrapped"


def install() -> None:
    """Patch source-free probability prediction row-count validation."""

    source_free = importlib.import_module("neureptrace.decoding.source_free")
    original_predict_source_probabilities = source_free._predict_source_probabilities
    if getattr(original_predict_source_probabilities, _FUNC_MARKER, False):
        setattr(source_free, _PATCH_MARKER, True)
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


__all__ = ["install"]
