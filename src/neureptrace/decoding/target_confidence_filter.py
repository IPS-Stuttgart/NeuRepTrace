"""Unlabeled target confidence filtering.

This module selects high-confidence target probability rows and optionally emits
pseudo-labels in a supplied source-class order.  It is a Category-2 helper:
target probability rows may be used for adaptation, but target labels are not part
of the public API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TARGET_CONFIDENCE_FILTER_PROTOCOL = "unlabeled_target_confidence_filter"
TARGET_CONFIDENCE_FILTER_CATEGORY = "2_unlabeled_target_adaptive"
SORT_MODES = ("none", "confidence", "entropy")
DEFAULT_MIN_CONFIDENCE = 0.8
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class TargetConfidenceFilterConfig:
    """Configuration for target probability confidence filtering."""

    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    max_entropy: float | None = None
    top_k: int | None = None
    sort_by: str = "none"
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class TargetConfidenceFilterResult:
    """Selected target rows and pseudo-label provenance."""

    selected_mask: np.ndarray
    selected_indices: np.ndarray
    probabilities: np.ndarray
    selected_probabilities: np.ndarray
    pseudo_label_indices: np.ndarray
    selected_pseudo_label_indices: np.ndarray
    pseudo_labels: np.ndarray | None
    selected_pseudo_labels: np.ndarray | None
    confidence: np.ndarray
    entropy: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-locals

def filter_target_probabilities_by_confidence(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    classes: Sequence[Any] | np.ndarray | None = None,
    config: TargetConfidenceFilterConfig | Mapping[str, Any] | None = None,
) -> TargetConfidenceFilterResult:
    """Select high-confidence target probability rows without target labels.

    Parameters
    ----------
    probabilities:
        Target-batch probability rows from a source-trained model. Rows are
        normalized internally.
    classes:
        Optional source-class order used to emit pseudo-label values. If omitted,
        only pseudo-label column indices are returned.
    config:
        Confidence-filter settings. A mapping is normalized through
        :func:`target_confidence_filter_config`.
    """

    cfg = target_confidence_filter_config() if config is None else _coerce_config(config)
    prob = _probability_matrix(probabilities, name="probabilities", epsilon=cfg.epsilon)
    class_values = None if classes is None else _classes(classes, n_classes=prob.shape[1])
    confidence = np.max(prob, axis=1)
    pseudo_indices = np.argmax(prob, axis=1).astype(int, copy=False)
    entropy = probability_entropy(prob, epsilon=cfg.epsilon)
    selected_mask = confidence >= cfg.min_confidence
    if cfg.max_entropy is not None:
        selected_mask &= entropy <= cfg.max_entropy
    if cfg.top_k is not None and np.count_nonzero(selected_mask) > cfg.top_k:
        selected_candidates = np.flatnonzero(selected_mask)
        order = _candidate_order(selected_candidates, confidence=confidence, entropy=entropy, sort_by=cfg.sort_by)
        keep = selected_candidates[order[: cfg.top_k]]
        selected_mask = np.zeros(prob.shape[0], dtype=bool)
        selected_mask[keep] = True
    selected_indices = _ordered_selected_indices(selected_mask, confidence=confidence, entropy=entropy, sort_by=cfg.sort_by)
    pseudo_labels = None if class_values is None else class_values[pseudo_indices]
    selected_pseudo_labels = None if pseudo_labels is None else pseudo_labels[selected_indices]
    metadata = _metadata(
        cfg,
        n_rows=prob.shape[0],
        n_classes=prob.shape[1],
        n_selected=selected_indices.shape[0],
        confidence=confidence,
        entropy=entropy,
    )
    return TargetConfidenceFilterResult(
        selected_mask=selected_mask,
        selected_indices=selected_indices.astype(int, copy=False),
        probabilities=prob.astype(np.float32, copy=False),
        selected_probabilities=prob[selected_indices].astype(np.float32, copy=False),
        pseudo_label_indices=pseudo_indices,
        selected_pseudo_label_indices=pseudo_indices[selected_indices],
        pseudo_labels=pseudo_labels,
        selected_pseudo_labels=selected_pseudo_labels,
        confidence=confidence.astype(np.float32, copy=False),
        entropy=entropy.astype(np.float32, copy=False),
        metadata=metadata,
    )


def target_confidence_filter_config(
    *,
    min_confidence: float | str = DEFAULT_MIN_CONFIDENCE,
    max_entropy: float | str | None = None,
    top_k: int | str | None = None,
    sort_by: str | None = "none",
    epsilon: float | str = DEFAULT_EPSILON,
) -> TargetConfidenceFilterConfig:
    """Normalize target-confidence filter options."""

    return TargetConfidenceFilterConfig(
        min_confidence=_unit_interval_float(min_confidence, name="min_confidence"),
        max_entropy=None if max_entropy in {None, "", "none", "None", "null"} else _nonnegative_float(max_entropy, name="max_entropy"),
        top_k=_optional_positive_int(top_k, name="top_k"),
        sort_by=normalize_sort_mode(sort_by),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def normalize_sort_mode(value: str | None) -> str:
    """Normalize selected-row ordering aliases."""

    normalized = "none" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"off": "none", "conf": "confidence", "max_probability": "confidence", "low_entropy": "entropy"}.get(normalized, normalized)
    if normalized not in SORT_MODES:
        raise ValueError(f"Unknown sort_by {value!r}. Available values: {', '.join(SORT_MODES)}.")
    return normalized


def probability_entropy(probabilities: Sequence[Sequence[float]] | np.ndarray, *, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    """Return row-wise entropy for probability rows."""

    prob = _probability_matrix(probabilities, name="probabilities", epsilon=epsilon)
    return -np.sum(prob * np.log(np.maximum(prob, _positive_float(epsilon, name="epsilon"))), axis=1)


def _coerce_config(config: TargetConfidenceFilterConfig | Mapping[str, Any]) -> TargetConfidenceFilterConfig:
    if isinstance(config, TargetConfidenceFilterConfig):
        return config
    return target_confidence_filter_config(**dict(config))


def _candidate_order(indices: np.ndarray, *, confidence: np.ndarray, entropy: np.ndarray, sort_by: str) -> np.ndarray:
    if sort_by == "confidence":
        return np.lexsort((indices, -confidence[indices]))
    if sort_by == "entropy":
        return np.lexsort((indices, entropy[indices]))
    return np.arange(indices.shape[0], dtype=int)


def _ordered_selected_indices(selected_mask: np.ndarray, *, confidence: np.ndarray, entropy: np.ndarray, sort_by: str) -> np.ndarray:
    indices = np.flatnonzero(selected_mask)
    if indices.shape[0] == 0:
        return indices
    return indices[_candidate_order(indices, confidence=confidence, entropy=entropy, sort_by=sort_by)]


def _metadata(
    cfg: TargetConfidenceFilterConfig,
    *,
    n_rows: int,
    n_classes: int,
    n_selected: int,
    confidence: np.ndarray,
    entropy: np.ndarray,
) -> dict[str, Any]:
    return {
        "target_confidence_filter": True,
        "target_confidence_filter_protocol": TARGET_CONFIDENCE_FILTER_PROTOCOL,
        "target_confidence_filter_protocol_category": TARGET_CONFIDENCE_FILTER_CATEGORY,
        "target_confidence_filter_uses_target_probabilities": True,
        "target_confidence_filter_uses_target_features": False,
        "target_confidence_filter_uses_target_labels": False,
        "target_confidence_filter_uses_target_pseudo_labels": True,
        "target_confidence_filter_valid_for_strict_source_only": False,
        "target_confidence_filter_valid_for_unlabeled_target_adaptation": True,
        "target_confidence_filter_valid_for_benchmark": False,
        "target_confidence_filter_n_rows": int(n_rows),
        "target_confidence_filter_n_classes": int(n_classes),
        "target_confidence_filter_n_selected": int(n_selected),
        "target_confidence_filter_min_confidence": float(cfg.min_confidence),
        "target_confidence_filter_max_entropy": "" if cfg.max_entropy is None else float(cfg.max_entropy),
        "target_confidence_filter_top_k": "" if cfg.top_k is None else int(cfg.top_k),
        "target_confidence_filter_sort_by": cfg.sort_by,
        "target_confidence_filter_mean_confidence": float(np.mean(confidence)),
        "target_confidence_filter_mean_entropy": float(np.mean(entropy)),
    }


def _classes(values: Sequence[Any] | np.ndarray, *, n_classes: int) -> np.ndarray:
    classes = np.asarray(values, dtype=object).reshape(-1)
    if classes.shape[0] != n_classes:
        raise ValueError(f"classes must contain one value per probability column: {classes.shape[0]} != {n_classes}.")
    if len(set(classes.tolist())) != classes.shape[0]:
        raise ValueError("classes must be unique.")
    return classes


def _probability_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str, epsilon: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix with at least two columns.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values.")
    matrix = np.maximum(matrix, _positive_float(epsilon, name="epsilon"))
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError(f"{name} rows must have positive mass.")
    return matrix / row_sums


def _optional_positive_int(value: int | str | None, *, name: str) -> int | None:
    if value in {None, "", "none", "None", "null"}:
        return None
    return _positive_int(value, name=name)


def _positive_int(value: int | str, *, name: str) -> int:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _unit_interval_float(value: float | str, *, name: str) -> float:
    parsed = _finite_float(value, name=name)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed


def _nonnegative_float(value: float | str, *, name: str) -> float:
    parsed = _finite_float(value, name=name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return parsed


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = _finite_float(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return parsed


def _finite_float(value: float | str, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed
