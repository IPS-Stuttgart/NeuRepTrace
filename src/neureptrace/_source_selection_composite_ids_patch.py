"""Preserve composite source-selection domain ids and labels."""

from __future__ import annotations

import importlib
from collections.abc import Hashable, Iterable, Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_selection_composite_ids_patch_installed"


def _object_value_vector(values: Iterable[Any]) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _object_vector_from_array(values: np.ndarray, *, name: str) -> np.ndarray:
    """Return a 1-D object vector while rejecting true matrix-shaped arrays."""

    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        return _object_value_vector([array.item()])
    if array.ndim == 1:
        return _object_value_vector(array.tolist())
    if array.ndim == 2 and 1 in array.shape:
        return _object_value_vector(array.reshape(-1).tolist())
    raise ValueError(f"{name} must be one-dimensional or a single-row/single-column vector; got shape {array.shape}.")


def _atomic_object_vector(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    """Return one object value per input row without flattening composites.

    Plain Python sequences such as ``[("subject", "run"), ...]`` are treated as
    one row value per outer-list item.  NumPy arrays are stricter: true matrices
    are rejected because they are usually malformed vector inputs, while
    single-row/single-column vectors remain accepted for CLI/config callers.
    """

    if isinstance(values, np.ndarray):
        return _object_vector_from_array(values, name=name)
    if isinstance(values, (str, bytes)):
        return _object_value_vector([values])
    try:
        items = list(values)
    except TypeError:
        items = [values]
    return _object_value_vector(items)


def _values_equal(left: Any, right: Any) -> bool:
    try:
        result = left == right
    except Exception:  # pragma: no cover - defensive fallback for unusual metadata objects
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    try:
        return bool(np.all(result))
    except Exception:  # pragma: no cover - defensive fallback for unusual metadata objects
        return False


def _object_equal_mask(vector: np.ndarray, value: Any) -> np.ndarray:
    """Compare object vectors item-wise so tuple values are atomic."""

    return np.asarray([_values_equal(item, value) for item in vector.tolist()], dtype=bool)


def _unique_values(vector: np.ndarray) -> tuple[Any, ...]:
    unique: list[Any] = []
    for value in vector.tolist():
        if not any(_values_equal(value, existing) for existing in unique):
            unique.append(value)
    return tuple(unique)


def _domain_vector(values: Sequence[Hashable] | np.ndarray, *, expected_length: int) -> np.ndarray:
    vector = _atomic_object_vector(values, name="source_domains")
    if vector.shape[0] != expected_length:
        raise ValueError(f"source_domains must contain one value per source row: {vector.shape[0]} != {expected_length}.")
    for domain in vector.tolist():
        try:
            hash(domain)
        except TypeError as exc:
            raise ValueError(f"source_domains must be hashable; got {domain!r}.") from exc
    return vector


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int) -> np.ndarray:
    vector = _atomic_object_vector(values, name="source_labels")
    if vector.shape[0] != expected_length:
        raise ValueError(f"source_labels must contain one value per source row: {vector.shape[0]} != {expected_length}.")
    return vector


def _class_balanced_weights(sample_weights: np.ndarray, source_labels: Sequence[Any] | np.ndarray, selected_mask: np.ndarray) -> np.ndarray:
    labels = _label_vector(source_labels, expected_length=sample_weights.shape[0])
    balanced = np.asarray(sample_weights, dtype=float).copy()
    selected_labels = _unique_values(labels[selected_mask])
    if not selected_labels:
        return balanced
    positive_mask = selected_mask & (balanced > 0.0)
    if not np.any(positive_mask):
        return balanced
    target_mass = float(np.sum(balanced[positive_mask]) / len(selected_labels))
    for label in selected_labels:
        class_mask = positive_mask & _object_equal_mask(labels, label)
        class_mass = float(np.sum(balanced[class_mask]))
        if class_mass > 0.0:
            balanced[class_mask] *= target_mass / class_mass
    return balanced


def install() -> None:
    """Patch source-domain selection to preserve tuple/list row identifiers."""

    source_selection = importlib.import_module("neureptrace.decoding.source_selection")
    original_select = source_selection.select_source_domains_by_target_similarity
    if getattr(original_select, _PATCH_MARKER, False):
        return

    @wraps(original_select)
    def select_source_domains_by_target_similarity(
        source_features: Sequence[Sequence[float]] | np.ndarray,
        source_domains: Sequence[Hashable] | np.ndarray,
        target_features: Sequence[Sequence[float]] | np.ndarray,
        *,
        metric: str | None = source_selection.DEFAULT_SOURCE_SELECTION_METRIC,
        top_k: int | str | None = None,
        max_distance: float | str | None = None,
        min_selected_domains: int | str = 1,
        softmax_temperature: float | str = source_selection.DEFAULT_SOURCE_SELECTION_TEMPERATURE,
        source_labels: Sequence[Any] | np.ndarray | None = None,
        class_balance: bool = False,
    ):
        source_matrix = source_selection._feature_matrix(source_features, name="source_features")
        target_matrix = source_selection._feature_matrix(target_features, name="target_features")
        if source_matrix.shape[1] != target_matrix.shape[1]:
            raise ValueError(
                "source_features and target_features must have the same feature width: "
                f"{source_matrix.shape[1]} != {target_matrix.shape[1]}."
            )
        domain_vector = _domain_vector(source_domains, expected_length=source_matrix.shape[0])
        domains = source_selection._unique_domains(domain_vector)
        selected_min = source_selection._normalize_positive_int(min_selected_domains, name="min_selected_domains")
        if selected_min > len(domains):
            raise ValueError(f"min_selected_domains={selected_min} exceeds the number of available source domains ({len(domains)}).")
        resolved_top_k = source_selection._normalize_optional_positive_int(top_k, name="top_k")
        if resolved_top_k is not None and resolved_top_k > len(domains):
            raise ValueError(f"top_k={resolved_top_k} exceeds the number of available source domains ({len(domains)}).")
        if resolved_top_k is not None and resolved_top_k < selected_min:
            raise ValueError("top_k must be greater than or equal to min_selected_domains.")
        resolved_max_distance = source_selection._normalize_optional_nonnegative_float(max_distance, name="max_distance")
        normalized_metric = source_selection.normalize_source_selection_metric(metric)

        distances = {
            domain: source_selection._domain_distance(source_matrix[_object_equal_mask(domain_vector, domain)], target_matrix, metric=normalized_metric)
            for domain in domains
        }
        ordered_domains = tuple(sorted(domains, key=lambda domain: (distances[domain], repr(domain))))
        selected = source_selection._select_domains(
            ordered_domains,
            distances,
            top_k=resolved_top_k,
            max_distance=resolved_max_distance,
            min_selected_domains=selected_min,
        )
        scores = source_selection._distance_scores(distances, temperature=softmax_temperature)
        selected_set = set(selected)
        selected_mask = np.asarray([domain in selected_set for domain in domain_vector.tolist()], dtype=bool)
        sample_weights = np.zeros(source_matrix.shape[0], dtype=float)
        for domain in selected:
            sample_weights[_object_equal_mask(domain_vector, domain)] = scores[domain]
        if class_balance:
            if source_labels is None:
                raise ValueError("source_labels are required when class_balance=True.")
            sample_weights = _class_balanced_weights(sample_weights, source_labels, selected_mask)
        sample_weights = source_selection._normalize_selected_weights(sample_weights, selected_mask)

        metadata = source_selection._metadata(
            metric=normalized_metric,
            n_source_rows=source_matrix.shape[0],
            n_target_rows=target_matrix.shape[0],
            feature_dim=source_matrix.shape[1],
            n_source_domains=len(domains),
            selected_domains=selected,
            top_k=resolved_top_k,
            max_distance=resolved_max_distance,
            min_selected_domains=selected_min,
            softmax_temperature=softmax_temperature,
            class_balance=class_balance,
            distances=distances,
            scores=scores,
        )
        return source_selection.SourceDomainSelectionResult(
            selected_domains=selected,
            domain_distances=distances,
            domain_scores=scores,
            sample_weights=sample_weights,
            selected_mask=selected_mask,
            metadata=metadata,
        )

    setattr(select_source_domains_by_target_similarity, _PATCH_MARKER, True)
    source_selection._domain_vector = _domain_vector
    source_selection._label_vector = _label_vector
    source_selection._class_balanced_weights = _class_balanced_weights
    source_selection.select_source_domains_by_target_similarity = select_source_domains_by_target_similarity


__all__ = ["install"]
