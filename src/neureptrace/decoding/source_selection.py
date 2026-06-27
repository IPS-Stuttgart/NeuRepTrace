"""Source-domain selection and weighting for cross-subject transfer.

This module implements a generic Category-2 source-domain selection helper that
can be used outside the MEKT-specific transfer path. Source subjects are scored
by similarity to an unlabeled held-out target feature distribution, then converted
into a selected-domain mask and optional sample weights. Target labels are not
accepted by the public API.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_DOMAIN_SELECTION_PROTOCOL = "unlabeled_target_source_domain_selection"
SOURCE_DOMAIN_SELECTION_CATEGORY = "2_unlabeled_target_adaptive"
SOURCE_DOMAIN_SELECTION_METRICS = ("mean", "covariance", "mean_covariance", "mmd")
DEFAULT_SOURCE_SELECTION_METRIC = "mean_covariance"
DEFAULT_SOURCE_SELECTION_TEMPERATURE = "auto"
_MIN_SCALE = 1e-12


@dataclass(frozen=True, slots=True)
class SourceDomainSelectionResult:
    """Target-similarity source-domain selection result."""

    selected_domains: tuple[Hashable, ...]
    domain_distances: Mapping[Hashable, float]
    domain_scores: Mapping[Hashable, float]
    sample_weights: np.ndarray
    selected_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments,too-many-locals
def select_source_domains_by_target_similarity(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_domains: Sequence[Hashable] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    metric: str | None = DEFAULT_SOURCE_SELECTION_METRIC,
    top_k: int | str | None = None,
    max_distance: float | str | None = None,
    min_selected_domains: int | str = 1,
    softmax_temperature: float | str = DEFAULT_SOURCE_SELECTION_TEMPERATURE,
    source_labels: Sequence[Any] | np.ndarray | None = None,
    class_balance: bool = False,
) -> SourceDomainSelectionResult:
    """Select or weight source domains by similarity to unlabeled target features."""

    source_matrix = _feature_matrix(source_features, name="source_features")
    target_matrix = _feature_matrix(target_features, name="target_features")
    if source_matrix.shape[1] != target_matrix.shape[1]:
        raise ValueError(
            "source_features and target_features must have the same feature width: "
            f"{source_matrix.shape[1]} != {target_matrix.shape[1]}."
        )

    domain_vector = _domain_vector(source_domains, expected_length=source_matrix.shape[0])
    domains = _unique_domains(domain_vector)
    selected_min = _normalize_positive_int(min_selected_domains, name="min_selected_domains")
    if selected_min > len(domains):
        raise ValueError(f"min_selected_domains={selected_min} exceeds the number of available source domains ({len(domains)}).")

    resolved_top_k = _normalize_optional_positive_int(top_k, name="top_k")
    if resolved_top_k is not None and resolved_top_k > len(domains):
        raise ValueError(f"top_k={resolved_top_k} exceeds the number of available source domains ({len(domains)}).")
    if resolved_top_k is not None and resolved_top_k < selected_min:
        raise ValueError("top_k must be greater than or equal to min_selected_domains.")

    resolved_max_distance = _normalize_optional_nonnegative_float(max_distance, name="max_distance")
    normalized_metric = normalize_source_selection_metric(metric)
    distances = {
        domain: _domain_distance(source_matrix[_object_equal_mask(domain_vector, domain)], target_matrix, metric=normalized_metric)
        for domain in domains
    }
    ordered_domains = tuple(sorted(domains, key=lambda domain: (distances[domain], repr(domain))))
    selected = _select_domains(
        ordered_domains,
        distances,
        top_k=resolved_top_k,
        max_distance=resolved_max_distance,
        min_selected_domains=selected_min,
    )
    scores = _distance_scores(distances, temperature=softmax_temperature)
    selected_set = set(selected)
    selected_mask = np.asarray([domain in selected_set for domain in domain_vector.tolist()], dtype=bool)
    sample_weights = np.zeros(source_matrix.shape[0], dtype=float)
    for domain in selected:
        sample_weights[_object_equal_mask(domain_vector, domain)] = scores[domain]
    if class_balance:
        if source_labels is None:
            raise ValueError("source_labels are required when class_balance=True.")
        sample_weights = _class_balanced_weights(sample_weights, source_labels, selected_mask)
    sample_weights = _normalize_selected_weights(sample_weights, selected_mask)

    metadata = _metadata(
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
    return SourceDomainSelectionResult(
        selected_domains=selected,
        domain_distances=distances,
        domain_scores=scores,
        sample_weights=sample_weights,
        selected_mask=selected_mask,
        metadata=metadata,
    )


def normalize_source_selection_metric(metric: str | None) -> str:
    """Normalize public aliases for target-similarity metrics."""

    normalized = DEFAULT_SOURCE_SELECTION_METRIC if metric is None else str(metric).strip().lower().replace("-", "_")
    normalized = {
        "mean_distance": "mean",
        "centroid": "mean",
        "centroid_distance": "mean",
        "cov": "covariance",
        "covariance_distance": "covariance",
        "coral": "covariance",
        "mean_cov": "mean_covariance",
        "mean_plus_covariance": "mean_covariance",
        "mean_coral": "mean_covariance",
        "mmd_rbf": "mmd",
        "rbf_mmd": "mmd",
        "maximum_mean_discrepancy": "mmd",
    }.get(normalized, normalized)
    if normalized not in SOURCE_DOMAIN_SELECTION_METRICS:
        raise ValueError(
            f"Unknown source-selection metric {metric!r}. "
            f"Available metrics: {', '.join(SOURCE_DOMAIN_SELECTION_METRICS)}."
        )
    return normalized


def selected_source_subset(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    result: SourceDomainSelectionResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return selected source rows, labels, and normalized non-zero weights."""

    features = _feature_matrix(source_features, name="source_features")
    labels = _label_vector(source_labels, expected_length=features.shape[0])
    mask = np.asarray(result.selected_mask, dtype=bool)
    if mask.shape[0] != features.shape[0]:
        raise ValueError("result.selected_mask length does not match source_features rows.")
    weights = np.asarray(result.sample_weights, dtype=float).reshape(-1)
    if weights.shape[0] != features.shape[0]:
        raise ValueError("result.sample_weights length does not match source_features rows.")
    return features[mask], labels[mask], weights[mask]


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one feature column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _domain_vector(values: Sequence[Hashable] | np.ndarray, *, expected_length: int) -> np.ndarray:
    vector = _object_vector(values, name="source_domains")
    if vector.shape[0] != expected_length:
        raise ValueError(f"source_domains must contain one value per source row: {vector.shape[0]} != {expected_length}.")
    for domain in vector.tolist():
        try:
            hash(domain)
        except TypeError as exc:
            raise ValueError(f"source_domains must be hashable; got {domain!r}.") from exc
    return vector


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int) -> np.ndarray:
    vector = _object_vector(values, name="source_labels")
    if vector.shape[0] != expected_length:
        raise ValueError(f"source_labels must contain one value per source row: {vector.shape[0]} != {expected_length}.")
    return vector


def _object_vector(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    """Return a one-dimensional object vector without flattening composite IDs.

    A Python list of tuples should represent tuple-valued IDs, while a genuine
    two-dimensional NumPy object matrix is ambiguous and must be rejected except
    for single-row/single-column vectors.
    """

    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            return array.reshape(1)
        if array.ndim == 1:
            return array.reshape(-1)
        if array.ndim == 2 and 1 in array.shape:
            return array.reshape(-1)
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")

    items = list(values)
    vector = np.empty(len(items), dtype=object)
    vector[:] = items
    return vector


def _object_equal_mask(values: np.ndarray, expected: Any) -> np.ndarray:
    return np.asarray([value == expected for value in values.tolist()], dtype=bool)


def _unique_domains(domain_vector: np.ndarray) -> tuple[Hashable, ...]:
    domains = tuple(dict.fromkeys(domain_vector.tolist()))
    if not domains:
        raise ValueError("At least one source domain is required.")
    return domains


def _domain_distance(source: np.ndarray, target: np.ndarray, *, metric: str) -> float:
    if source.shape[0] < 1:
        raise ValueError("Each source domain must contain at least one source row.")
    mean_distance = float(np.linalg.norm(np.mean(source, axis=0) - np.mean(target, axis=0)) / np.sqrt(source.shape[1]))
    if metric == "mean":
        return mean_distance
    if metric in {"covariance", "mean_covariance"}:
        covariance_distance = _covariance_distance(source, target)
        return covariance_distance if metric == "covariance" else mean_distance + covariance_distance
    if metric == "mmd":
        return _mmd_distance(source, target)
    raise ValueError(f"Unhandled source-selection metric {metric!r}.")


def _covariance_distance(source: np.ndarray, target: np.ndarray) -> float:
    source_cov = _covariance(source)
    target_cov = _covariance(target)
    return float(np.linalg.norm(source_cov - target_cov, ord="fro") / max(1, source.shape[1]))


def _covariance(values: np.ndarray) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros((values.shape[1], values.shape[1]), dtype=float)
    centered = values - np.mean(values, axis=0, keepdims=True)
    return centered.T @ centered / float(values.shape[0] - 1)


def _mmd_distance(source: np.ndarray, target: np.ndarray) -> float:
    combined = np.vstack([source, target])
    sigma2 = _median_squared_distance(combined)
    gamma = 1.0 / (2.0 * max(sigma2, _MIN_SCALE))
    k_xx = _rbf_kernel(source, source, gamma)
    k_yy = _rbf_kernel(target, target, gamma)
    k_xy = _rbf_kernel(source, target, gamma)
    mmd2 = float(np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy))
    return float(np.sqrt(max(0.0, mmd2)))


def _median_squared_distance(values: np.ndarray) -> float:
    if values.shape[0] <= 1:
        return 1.0
    squared = _squared_euclidean(values, values)
    upper = squared[np.triu_indices(values.shape[0], k=1)]
    positive = upper[upper > 0.0]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def _rbf_kernel(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * _squared_euclidean(left, right))


def _squared_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    return np.maximum(left_norm + right_norm - 2.0 * (left @ right.T), 0.0)


def _select_domains(
    ordered_domains: tuple[Hashable, ...],
    distances: Mapping[Hashable, float],
    *,
    top_k: int | None,
    max_distance: float | None,
    min_selected_domains: int,
) -> tuple[Hashable, ...]:
    if max_distance is None:
        selected = list(ordered_domains)
    else:
        selected = [domain for domain in ordered_domains if distances[domain] <= max_distance]
    if top_k is not None:
        selected = selected[:top_k]
    for domain in ordered_domains:
        if len(selected) >= min_selected_domains:
            break
        if domain not in selected:
            selected.append(domain)
    return tuple(selected)


def _distance_scores(distances: Mapping[Hashable, float], *, temperature: float | str) -> dict[Hashable, float]:
    values = np.asarray([float(value) for value in distances.values()], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("All source-domain distances must be finite.")
    min_distance = float(np.min(values))
    gaps = values - min_distance
    resolved_temperature = _resolve_temperature(gaps, temperature)
    scores = np.exp(-gaps / resolved_temperature)
    return {domain: float(score) for domain, score in zip(distances.keys(), scores, strict=True)}


def _resolve_temperature(distance_gaps: np.ndarray, temperature: float | str) -> float:
    if isinstance(temperature, str):
        normalized = temperature.strip().lower()
        if normalized == "auto":
            positive = distance_gaps[distance_gaps > _MIN_SCALE]
            return float(max(np.median(positive), _MIN_SCALE)) if positive.size else 1.0
        parsed = float(normalized)
    else:
        parsed = float(temperature)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError("softmax_temperature must be a positive finite value or 'auto'.")
    return parsed


def _class_balanced_weights(sample_weights: np.ndarray, source_labels: Sequence[Any] | np.ndarray, selected_mask: np.ndarray) -> np.ndarray:
    labels = _label_vector(source_labels, expected_length=sample_weights.shape[0])
    balanced = np.asarray(sample_weights, dtype=float).copy()
    selected_labels = tuple(dict.fromkeys(labels[selected_mask].tolist()))
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


def _normalize_selected_weights(sample_weights: np.ndarray, selected_mask: np.ndarray) -> np.ndarray:
    weights = np.asarray(sample_weights, dtype=float).reshape(-1).copy()
    if weights.shape[0] != selected_mask.shape[0]:
        raise ValueError("sample_weights and selected_mask must have the same length.")
    weights[~selected_mask] = 0.0
    if not np.any(selected_mask):
        raise ValueError("At least one source row must be selected.")
    total = float(np.sum(weights[selected_mask]))
    if total <= 0.0:
        weights[selected_mask] = 1.0
        total = float(np.sum(weights[selected_mask]))
    weights[selected_mask] *= float(np.count_nonzero(selected_mask)) / total
    return weights


def _metadata(
    *,
    metric: str,
    n_source_rows: int,
    n_target_rows: int,
    feature_dim: int,
    n_source_domains: int,
    selected_domains: tuple[Hashable, ...],
    top_k: int | None,
    max_distance: float | None,
    min_selected_domains: int,
    softmax_temperature: float | str,
    class_balance: bool,
    distances: Mapping[Hashable, float],
    scores: Mapping[Hashable, float],
) -> dict[str, Any]:
    return {
        "source_domain_selection": True,
        "source_selection_protocol": SOURCE_DOMAIN_SELECTION_PROTOCOL,
        "source_selection_protocol_category": SOURCE_DOMAIN_SELECTION_CATEGORY,
        "source_selection_metric": metric,
        "source_selection_uses_source_features": True,
        "source_selection_uses_source_domains": True,
        "source_selection_uses_source_labels": bool(class_balance),
        "source_selection_uses_target_features": True,
        "source_selection_uses_target_labels": False,
        "source_selection_valid_for_strict_source_only": False,
        "source_selection_valid_for_unlabeled_target_adaptation": True,
        "source_selection_valid_for_benchmark": False,
        "source_selection_n_source_rows": int(n_source_rows),
        "source_selection_n_target_rows": int(n_target_rows),
        "source_selection_feature_dim": int(feature_dim),
        "source_selection_n_source_domains": int(n_source_domains),
        "source_selection_n_selected_domains": int(len(selected_domains)),
        "source_selection_selected_domains": "|".join(str(domain) for domain in selected_domains),
        "source_selection_top_k": "" if top_k is None else int(top_k),
        "source_selection_max_distance": "" if max_distance is None else float(max_distance),
        "source_selection_min_selected_domains": int(min_selected_domains),
        "source_selection_softmax_temperature": str(softmax_temperature),
        "source_selection_class_balance": bool(class_balance),
        "source_selection_domain_distances": _format_mapping(distances),
        "source_selection_domain_scores": _format_mapping(scores),
    }


def _format_mapping(values: Mapping[Hashable, float]) -> str:
    return "|".join(f"{str(domain)}:{float(value):.12g}" for domain, value in values.items())


def _normalize_positive_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0 or numeric < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(numeric)


def _normalize_optional_positive_int(value: int | str | None, *, name: str) -> int | None:
    if value in {None, "", "none", "None", "null", "all", "full"}:
        return None
    return _normalize_positive_int(value, name=name)


def _normalize_optional_nonnegative_float(value: float | str | None, *, name: str) -> float | None:
    if value in {None, "", "none", "None", "null", "off"}:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return parsed
