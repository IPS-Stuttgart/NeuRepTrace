"""Install temporal-model validation, real-valued input checks, and exact-k tie handling."""

from __future__ import annotations

import importlib
from functools import wraps

import numpy as np

from . import _temporal_model_baseline_duplicate_patch, _temporal_model_class_metadata_patch

_PATCH_MARKER = "_neureptrace_temporal_smoothing_topk_tie_patch_installed"


def _contains_complex_values(values: object) -> bool:
    """Return whether an array-like object contains Python/NumPy complex scalars."""

    array = np.asarray(values, dtype=object)
    if array.size == 0:
        return False
    return any(isinstance(value, (complex, np.complexfloating)) for value in array.ravel())


def _reject_complex_values(values: object, message: str) -> None:
    if _contains_complex_values(values):
        raise ValueError(message)


def _validate_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0 or numeric < 1.0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(numeric)


def _stable_top_columns(probabilities: np.ndarray, *, k: int) -> np.ndarray:
    _reject_complex_values(
        probabilities,
        "probabilities must contain real-valued probabilities, not complex values.",
    )
    probability_matrix = np.asarray(probabilities, dtype=float)
    if probability_matrix.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional matrix.")
    effective_k = min(_validate_positive_integer(k, name="k"), probability_matrix.shape[1])
    return np.argsort(-probability_matrix, axis=1, kind="mergesort")[:, :effective_k]


def _top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, *, k: int) -> float:
    """Compute exact-k top-k accuracy with stable class-index tie handling."""

    _reject_complex_values(
        labels,
        "labels must contain real-valued class indices, not complex values.",
    )
    label_indices = np.asarray(labels, dtype=int).reshape(-1)
    if label_indices.size == 0:
        return float("nan")
    top_columns = _stable_top_columns(probabilities, k=k)
    if top_columns.shape[0] != label_indices.size:
        raise ValueError("labels must have one entry per probability row.")
    return float(np.mean(np.any(top_columns == label_indices[:, None], axis=1)))


def _top_k_accuracy_from_label_values(probabilities: np.ndarray, labels: np.ndarray, label_values: tuple[int, ...], *, k: int) -> float:
    """Compute exact-k top-k accuracy for arbitrary integer class labels."""

    _reject_complex_values(
        labels,
        "labels must contain real-valued class indices, not complex values.",
    )
    label_array = np.asarray(labels, dtype=int).reshape(-1)
    if label_array.size == 0:
        return float("nan")
    top_columns = _stable_top_columns(probabilities, k=k)
    if top_columns.shape[0] != label_array.size:
        raise ValueError("labels must have one entry per probability row.")
    top_labels = np.asarray(label_values, dtype=int)[top_columns]
    return float(np.mean(np.any(top_labels == label_array[:, None], axis=1)))


def install() -> None:
    """Install temporal-model validation, real-valued inputs, and stable exact-k metrics."""

    _temporal_model_baseline_duplicate_patch.install()
    _temporal_model_class_metadata_patch.install()

    temporal_smoothing = importlib.import_module("neureptrace.temporal_smoothing")
    if getattr(temporal_smoothing, _PATCH_MARKER, False):
        return

    original_numeric_label_values = temporal_smoothing._numeric_label_values

    @wraps(original_numeric_label_values)
    def _numeric_label_values(frame, label_values):
        if "true_label" in frame.columns:
            _reject_complex_values(
                frame["true_label"].to_numpy(dtype=object),
                "true_label values must be real-valued integer labels, not complex values.",
            )
        return original_numeric_label_values(frame, label_values)

    original_metrics_from_probability_observations = temporal_smoothing.metrics_from_probability_observations

    @wraps(original_metrics_from_probability_observations)
    def metrics_from_probability_observations(observations, *, ece_bins: int = 10):
        for column in temporal_smoothing.probability_columns(observations):
            _reject_complex_values(
                observations[column].to_numpy(dtype=object),
                f"{column} values must be real-valued probabilities, not complex values.",
            )
        return original_metrics_from_probability_observations(observations, ece_bins=ece_bins)

    temporal_smoothing._numeric_label_values = _numeric_label_values
    temporal_smoothing.metrics_from_probability_observations = metrics_from_probability_observations
    temporal_smoothing._top_k_accuracy = _top_k_accuracy
    temporal_smoothing._top_k_accuracy_from_label_values = _top_k_accuracy_from_label_values
    setattr(temporal_smoothing, _PATCH_MARKER, True)


__all__ = ["install"]
