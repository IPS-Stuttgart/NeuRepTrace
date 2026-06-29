"""Reject boolean inputs in probability-stacking numeric paths.

Probability stacking accepts labels, probability matrices, and metric tables from
CSV-like sources where pandas and NumPy otherwise coerce booleans to integer
``0``/``1`` values.  Booleans are malformed probability/label data in these
paths, so reject them before the core numeric validators run.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

from neureptrace.observations import probability_columns

_PATCH_MARKER = "_neureptrace_probability_stacking_bool_validation_patch_installed"


def _contains_boolean_values(values: Any) -> bool:
    """Return whether an array-like value contains Python or NumPy booleans."""

    if isinstance(values, pd.DataFrame):
        return any(_contains_boolean_values(values[column]) for column in values.columns)
    if isinstance(values, pd.Series):
        if pd.api.types.is_bool_dtype(values.dtype):
            return True
        if values.dtype == object:
            return bool(values.map(lambda value: isinstance(value, (bool, np.bool_))).any())
        return False

    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.bool_):
        return True
    if array.dtype == object:
        return any(isinstance(value, (bool, np.bool_)) for value in array.ravel())
    return False


def _reject_boolean_values(values: Any, *, message: str) -> None:
    if _contains_boolean_values(values):
        raise ValueError(message)


def _reject_boolean_label_values(values: Any, *, name: str) -> None:
    _reject_boolean_values(values, message=f"{name} values must be numeric, not boolean.")


def _reject_boolean_probabilities(values: Any, *, context: str = "Probability values") -> None:
    _reject_boolean_values(values, message=f"{context} must be numeric, not boolean.")


def _reject_boolean_probability_columns(observations: pd.DataFrame, *, context: str = "Probability values") -> None:
    prob_columns = probability_columns(observations)
    if prob_columns:
        _reject_boolean_probabilities(observations.loc[:, list(prob_columns)], context=context)


def install() -> None:
    """Install boolean guards on probability-stacking public numeric paths."""

    from neureptrace import probability_stacking as ps

    if getattr(ps, _PATCH_MARKER, False):
        return

    original_integer_label_array = ps._integer_label_array

    @wraps(original_integer_label_array)
    def _integer_label_array(labels: Any, *, name: str):
        _reject_boolean_label_values(labels, name=name)
        return original_integer_label_array(labels, name=name)

    original_renormalize_probabilities = ps._renormalize_probabilities

    @wraps(original_renormalize_probabilities)
    def _renormalize_probabilities(
        values: Any,
        *,
        min_probability: float = ps.DEFAULT_MIN_PROBABILITY,
        require_normalized: bool = True,
    ):
        _reject_boolean_probabilities(values)
        return original_renormalize_probabilities(
            values,
            min_probability=min_probability,
            require_normalized=require_normalized,
        )

    original_validate_probability_matrix = ps._validate_probability_matrix

    @wraps(original_validate_probability_matrix)
    def _validate_probability_matrix(values: Any, *, context: str = "Probability values"):
        _reject_boolean_probabilities(values, context=context)
        return original_validate_probability_matrix(values, context=context)

    original_top_k_accuracy = ps._top_k_accuracy

    @wraps(original_top_k_accuracy)
    def _top_k_accuracy(probabilities: Any, labels: Any, *, k: int):
        _reject_boolean_probabilities(probabilities)
        _reject_boolean_label_values(labels, name="labels")
        return original_top_k_accuracy(probabilities, labels, k=k)

    original_top_k_accuracy_from_label_values = ps._top_k_accuracy_from_label_values

    @wraps(original_top_k_accuracy_from_label_values)
    def _top_k_accuracy_from_label_values(probabilities: Any, true_labels: Any, label_values: Any, *, k: int):
        _reject_boolean_probabilities(probabilities)
        _reject_boolean_label_values(true_labels, name="true_labels")
        return original_top_k_accuracy_from_label_values(probabilities, true_labels, label_values, k=k)

    original_align_probability_cube = ps.align_probability_cube

    @wraps(original_align_probability_cube)
    def align_probability_cube(observations: pd.DataFrame, *args: Any, **kwargs: Any):
        _reject_boolean_probability_columns(observations)
        return original_align_probability_cube(observations, *args, **kwargs)

    original_fit_stacking_weights = ps.fit_stacking_weights

    @wraps(original_fit_stacking_weights)
    def fit_stacking_weights(probability_cube: Any, labels: Any, *args: Any, **kwargs: Any):
        _reject_boolean_probabilities(probability_cube)
        _reject_boolean_label_values(labels, name="labels")
        return original_fit_stacking_weights(probability_cube, labels, *args, **kwargs)

    original_fit_source_oof_stacking = ps.fit_source_oof_stacking

    @wraps(original_fit_source_oof_stacking)
    def fit_source_oof_stacking(source_probability_cube: Any, source_labels: Any, *args: Any, **kwargs: Any):
        _reject_boolean_probabilities(source_probability_cube)
        _reject_boolean_label_values(source_labels, name="source_labels")
        return original_fit_source_oof_stacking(source_probability_cube, source_labels, *args, **kwargs)

    original_combine_probability_cube = ps.combine_probability_cube

    @wraps(original_combine_probability_cube)
    def combine_probability_cube(probability_cube: Any, weights: Any, *args: Any, **kwargs: Any):
        _reject_boolean_probabilities(probability_cube)
        _reject_boolean_values(weights, message="weights must be numeric, not boolean.")
        return original_combine_probability_cube(probability_cube, weights, *args, **kwargs)

    original_summarize_stacked_metrics = ps.summarize_stacked_metrics

    @wraps(original_summarize_stacked_metrics)
    def summarize_stacked_metrics(observations: pd.DataFrame):
        prob_columns = probability_columns(observations)
        if prob_columns:
            _reject_boolean_probability_columns(observations, context="Probability values")
            if "true_label" in observations.columns:
                _reject_boolean_label_values(observations["true_label"], name="true_label")
        return original_summarize_stacked_metrics(observations)

    ps._integer_label_array = _integer_label_array
    ps._renormalize_probabilities = _renormalize_probabilities
    ps._validate_probability_matrix = _validate_probability_matrix
    ps._top_k_accuracy = _top_k_accuracy
    ps._top_k_accuracy_from_label_values = _top_k_accuracy_from_label_values
    ps.align_probability_cube = align_probability_cube
    ps.fit_stacking_weights = fit_stacking_weights
    ps.fit_source_oof_stacking = fit_source_oof_stacking
    ps.combine_probability_cube = combine_probability_cube
    ps.summarize_stacked_metrics = summarize_stacked_metrics
    setattr(ps, _PATCH_MARKER, True)


__all__ = ["install"]