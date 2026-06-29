"""Reject boolean numeric values in response-window observation inputs.

Pandas and NumPy treat booleans as numeric during coercion. For response-window
ensembles this can silently turn malformed ``true_label`` or ``prob_class_*``
values into valid-looking 0/1 labels and probabilities.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

_PATCH_MARKER = "_neureptrace_response_window_bool_numeric_patch_installed"


def _is_bool_scalar(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _contains_boolean(values: object) -> bool:
    array = np.asarray(values)
    if array.dtype.kind == "b":
        return bool(array.size)
    if array.dtype == object:
        object_array = np.asarray(values, dtype=object)
        return any(_is_bool_scalar(value) for value in object_array.ravel())
    return False


def _boolean_mask(values: pd.Series) -> pd.Series:
    return values.map(_is_bool_scalar).fillna(False).astype(bool)


def install() -> None:
    """Patch response-window numeric validation to reject booleans."""

    temporal_model = importlib.import_module("neureptrace.temporal_model")
    response_window_ensemble = importlib.import_module("neureptrace.response_window_ensemble")

    if getattr(temporal_model, _PATCH_MARKER, False) and getattr(response_window_ensemble, _PATCH_MARKER, False):
        return

    if not getattr(temporal_model, _PATCH_MARKER, False):
        original_validate_probability_matrix = temporal_model._validate_probability_matrix

        @wraps(original_validate_probability_matrix)
        def _validate_probability_matrix(probabilities: np.ndarray) -> np.ndarray:
            if _contains_boolean(probabilities):
                raise ValueError("Probability observations must be numeric probabilities, not booleans.")
            return original_validate_probability_matrix(probabilities)

        temporal_model._validate_probability_matrix = _validate_probability_matrix
        setattr(temporal_model, _PATCH_MARKER, True)

    if not getattr(response_window_ensemble, _PATCH_MARKER, False):
        original_integer_label_values = response_window_ensemble._integer_label_values

        @wraps(original_integer_label_values)
        def _integer_label_values(values: Sequence[object] | np.ndarray | pd.Series, *, n_classes: int | None = None) -> np.ndarray:
            series = pd.Series(values)
            boolean_values = _boolean_mask(series)
            if bool(boolean_values.any()):
                examples = [repr(value) for value in series.loc[boolean_values].head(5).tolist()]
                raise ValueError(
                    "Response-window true_label values must be numeric integer labels, not booleans; "
                    f"invalid values: {examples}"
                )
            return original_integer_label_values(values, n_classes=n_classes)

        response_window_ensemble._integer_label_values = _integer_label_values
        setattr(response_window_ensemble, _PATCH_MARKER, True)


__all__ = ["install"]
