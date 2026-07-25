"""Oracle target-prior probability adapter.

This module is a Category-4 diagnostic helper.  It estimates the class prior from
held-out target labels and uses that prior to rescale probability rows.  Because
it uses scored target labels, it is an oracle/debug upper bound and must not be
reported as a benchmark-valid deployment protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neureptrace._object_label_utils import label_counts, label_equal_mask, values_equal

ORACLE_TARGET_PRIOR_PROTOCOL = "oracle_target_label_prior_adapter"
ORACLE_TARGET_PRIOR_CATEGORY = "4_oracle_upper_bound"
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class OracleTargetPriorResult:
    """Oracle prior-adapted probabilities and provenance metadata."""

    probabilities: np.ndarray
    original_probabilities: np.ndarray
    classes: np.ndarray
    source_prior: np.ndarray
    target_prior: np.ndarray
    prior_ratio: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def apply_oracle_target_prior(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    target_labels: Sequence[Any] | np.ndarray,
    *,
    classes: Sequence[Any] | np.ndarray | None = None,
    source_prior: Sequence[float] | np.ndarray | None = None,
    epsilon: float | str = DEFAULT_EPSILON,
) -> OracleTargetPriorResult:
    """Rescale probability rows using the true target-label prior.

    This function intentionally uses ``target_labels``.  It is therefore a
    Category-4 oracle/debug helper, not a benchmark-valid deployment method.
    """

    eps = _positive_float(epsilon, name="epsilon")
    probs = _probability_matrix(probabilities, name="probabilities", epsilon=eps)
    labels = _value_vector(target_labels, expected_length=probs.shape[0], name="target_labels")
    class_values = _classes(labels, classes=classes, n_columns=probs.shape[1])
    src_prior = _prior_vector(source_prior, n_classes=probs.shape[1], epsilon=eps)
    tgt_prior = oracle_target_prior(labels, classes=class_values, epsilon=eps)
    ratio = tgt_prior / src_prior
    adapted = _normalize_probability_rows(probs * ratio[None, :], epsilon=eps)
    metadata = {
        "oracle_target_prior": True,
        "oracle_target_prior_protocol": ORACLE_TARGET_PRIOR_PROTOCOL,
        "oracle_target_prior_protocol_category": ORACLE_TARGET_PRIOR_CATEGORY,
        "oracle_target_prior_uses_target_probabilities": True,
        "oracle_target_prior_uses_target_labels": True,
        "oracle_target_prior_valid_for_strict_source_only": False,
        "oracle_target_prior_valid_for_unlabeled_target_adaptation": False,
        "oracle_target_prior_valid_for_supervised_calibration": False,
        "oracle_target_prior_valid_for_benchmark": False,
        "oracle_target_prior_debug_upper_bound": True,
        "oracle_target_prior_n_rows": int(probs.shape[0]),
        "oracle_target_prior_n_classes": int(probs.shape[1]),
        "oracle_target_prior_epsilon": float(eps),
        "oracle_target_prior_source_prior": "|".join(f"{value:.12g}" for value in src_prior.tolist()),
        "oracle_target_prior_target_prior": "|".join(f"{value:.12g}" for value in tgt_prior.tolist()),
    }
    return OracleTargetPriorResult(
        probabilities=adapted.astype(np.float32, copy=False),
        original_probabilities=probs.astype(np.float32, copy=False),
        classes=class_values,
        source_prior=src_prior.astype(np.float32, copy=False),
        target_prior=tgt_prior.astype(np.float32, copy=False),
        prior_ratio=ratio.astype(np.float32, copy=False),
        metadata=metadata,
    )


def oracle_target_prior(target_labels: Sequence[Any] | np.ndarray, *, classes: Sequence[Any] | np.ndarray, epsilon: float | str = DEFAULT_EPSILON) -> np.ndarray:
    """Return the true target-label prior in the requested class order."""

    eps = _positive_float(epsilon, name="epsilon")
    labels = _value_vector(target_labels, name="target_labels")
    class_values = _value_vector(classes, name="classes")
    if labels.shape[0] < 1:
        raise ValueError("target_labels must contain at least one value.")
    if class_values.shape[0] < 2:
        raise ValueError("classes must contain at least two values.")
    _validate_unique_classes(class_values)
    unknown = _missing_labels(labels, class_values)
    if unknown:
        raise ValueError(f"target_labels contain labels absent from classes: {unknown}.")
    counts = np.asarray([np.count_nonzero(label_equal_mask(labels, class_label)) for class_label in class_values.tolist()], dtype=float)
    return _normalize_probability_rows(counts[None, :], epsilon=eps)[0]


def _classes(labels: np.ndarray, *, classes: Sequence[Any] | np.ndarray | None, n_columns: int) -> np.ndarray:
    if classes is None:
        values, _counts = label_counts(labels)
    else:
        values = _value_vector(classes, expected_length=n_columns, name="classes")
    if values.shape[0] != n_columns:
        raise ValueError(f"classes must contain one value per probability column: {values.shape[0]} != {n_columns}.")
    _validate_unique_classes(values)
    unknown = _missing_labels(labels, values)
    if unknown:
        raise ValueError(f"target_labels contain labels absent from classes: {unknown}.")
    return values


def _prior_vector(values: Sequence[float] | np.ndarray | None, *, n_classes: int, epsilon: float) -> np.ndarray:
    if values is None:
        prior = np.full(n_classes, 1.0 / n_classes, dtype=float)
    else:
        prior = np.asarray(values, dtype=float).reshape(-1)
        if prior.shape[0] != n_classes:
            raise ValueError(f"source_prior must contain one value per probability column: {prior.shape[0]} != {n_classes}.")
    return _normalize_probability_rows(prior[None, :], epsilon=epsilon)[0]


def _probability_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str, epsilon: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix with at least two columns.")
    return _normalize_probability_rows(matrix, epsilon=epsilon)


def _normalize_probability_rows(values: np.ndarray, *, epsilon: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probabilities must be finite and non-negative.")
    matrix = np.maximum(matrix, float(epsilon))
    row_maxima = np.max(matrix, axis=1, keepdims=True)
    scaled = matrix / row_maxima
    row_sums = np.sum(scaled, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must have positive mass.")
    return scaled / row_sums


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _object_vector(values: Sequence[Any]) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _value_vector(values: Sequence[Any] | np.ndarray, *, name: str, expected_length: int | None = None) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        items = [values]
    else:
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            items = [array.item()]
        elif array.ndim == 1:
            if expected_length is not None and array.shape[0] != expected_length and expected_length == 1:
                items = [tuple(array.tolist())]
            else:
                items = array.reshape(-1).tolist()
        else:
            rows = array.reshape(array.shape[0], -1)
            if rows.shape[1] == 1:
                items = rows[:, 0].tolist()
            else:
                items = [tuple(row.tolist()) for row in rows]

    if expected_length is not None and len(items) != expected_length:
        unit = "probability row" if name == "target_labels" else "probability column"
        raise ValueError(f"{name} must contain one value per {unit}: {len(items)} != {expected_length}.")
    return _object_vector(items)


def _validate_unique_classes(class_values: np.ndarray) -> None:
    unique, _counts = label_counts(class_values)
    if unique.shape[0] != class_values.shape[0]:
        raise ValueError("classes must be unique.")


def _missing_labels(labels: np.ndarray, class_values: np.ndarray) -> list[object]:
    unique_labels, _counts = label_counts(labels)
    classes_list = class_values.tolist()
    return sorted(
        [label for label in unique_labels.tolist() if not any(values_equal(label, class_label) for class_label in classes_list)],
        key=repr,
    )
