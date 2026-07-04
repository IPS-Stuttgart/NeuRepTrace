"""Unlabeled target confidence selection.

This module implements a small Category-2 selective-prediction helper.  It takes
class-probability rows for an unlabeled target batch, computes confidence and
margin scores, and returns a mask of rows that should be kept for high-confidence
reporting or downstream pseudo-labeling.  It intentionally has no target-label
argument.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TARGET_CONFIDENCE_SELECTION_PROTOCOL = "unlabeled_target_confidence_selection"
TARGET_CONFIDENCE_SELECTION_CATEGORY = "2_unlabeled_target_adaptive"
DEFAULT_MIN_CONFIDENCE = 0.0
DEFAULT_MIN_MARGIN = 0.0
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class TargetConfidenceSelectionConfig:
    """Configuration for unlabeled target confidence selection."""

    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    min_margin: float = DEFAULT_MIN_MARGIN
    top_fraction: float | None = None
    min_keep_rows: int = 0
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class TargetConfidenceSelectionResult:
    """Selected target rows, pseudo-labels, and protocol metadata."""

    probabilities: np.ndarray
    predictions: np.ndarray
    classes: np.ndarray
    confidence: np.ndarray
    margin: np.ndarray
    keep_mask: np.ndarray
    selected_indices: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-locals

def select_target_confident_predictions(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    classes: Sequence[Any] | np.ndarray | None = None,
    config: TargetConfidenceSelectionConfig | Mapping[str, Any] | None = None,
) -> TargetConfidenceSelectionResult:
    """Select confident rows from an unlabeled target probability batch.

    Parameters
    ----------
    probabilities:
        Target probability rows, typically from a source-trained model.  Rows are
        normalized internally.
    classes:
        Optional class labels corresponding to probability columns.  If omitted,
        integer column indices are used.
    config:
        Selection settings.  A mapping is normalized through
        :func:`target_confidence_selection_config`.

    Returns
    -------
    TargetConfidenceSelectionResult
        Normalized probabilities, pseudo-label predictions, confidence/margin
        scores, selected row indices, and protocol metadata.
    """

    cfg = target_confidence_selection_config() if config is None else _coerce_config(config)
    probs = _probability_matrix(probabilities, epsilon=cfg.epsilon)
    class_values = _classes(classes, n_classes=probs.shape[1])
    order = np.argsort(probs, axis=1)
    top = order[:, -1]
    second = order[:, -2]
    confidence = probs[np.arange(probs.shape[0]), top]
    margin = confidence - probs[np.arange(probs.shape[0]), second]
    predictions = class_values[top]
    keep_mask = (confidence >= cfg.min_confidence) & (margin >= cfg.min_margin)
    if cfg.top_fraction is not None:
        keep_mask &= _top_fraction_mask(confidence, top_fraction=cfg.top_fraction)
    if cfg.min_keep_rows > 0 and int(np.count_nonzero(keep_mask)) < cfg.min_keep_rows:
        keep_mask = _force_min_keep(confidence, keep_mask=keep_mask, min_keep_rows=cfg.min_keep_rows)
    selected_indices = np.flatnonzero(keep_mask)
    metadata = {
        "target_confidence_selection": True,
        "target_confidence_selection_protocol": TARGET_CONFIDENCE_SELECTION_PROTOCOL,
        "target_confidence_selection_protocol_category": TARGET_CONFIDENCE_SELECTION_CATEGORY,
        "target_confidence_selection_uses_target_probabilities": True,
        "target_confidence_selection_uses_target_features": False,
        "target_confidence_selection_uses_target_labels": False,
        "target_confidence_selection_valid_for_strict_source_only": False,
        "target_confidence_selection_valid_for_unlabeled_target_adaptation": True,
        "target_confidence_selection_valid_for_benchmark": False,
        "target_confidence_selection_n_rows": int(probs.shape[0]),
        "target_confidence_selection_n_classes": int(probs.shape[1]),
        "target_confidence_selection_n_selected_rows": int(selected_indices.shape[0]),
        "target_confidence_selection_min_confidence": float(cfg.min_confidence),
        "target_confidence_selection_min_margin": float(cfg.min_margin),
        "target_confidence_selection_top_fraction": "" if cfg.top_fraction is None else float(cfg.top_fraction),
        "target_confidence_selection_min_keep_rows": int(cfg.min_keep_rows),
        "target_confidence_selection_mean_confidence": float(np.mean(confidence)),
        "target_confidence_selection_mean_margin": float(np.mean(margin)),
    }
    return TargetConfidenceSelectionResult(
        probabilities=probs.astype(np.float32, copy=False),
        predictions=predictions,
        classes=class_values,
        confidence=confidence.astype(np.float32, copy=False),
        margin=margin.astype(np.float32, copy=False),
        keep_mask=keep_mask.astype(bool, copy=False),
        selected_indices=selected_indices.astype(int, copy=False),
        metadata=metadata,
    )


def target_confidence_selection_config(
    *,
    min_confidence: float | str = DEFAULT_MIN_CONFIDENCE,
    min_margin: float | str = DEFAULT_MIN_MARGIN,
    top_fraction: float | str | None = None,
    min_keep_rows: int | str = 0,
    epsilon: float | str = DEFAULT_EPSILON,
) -> TargetConfidenceSelectionConfig:
    """Normalize target-confidence selection options."""

    return TargetConfidenceSelectionConfig(
        min_confidence=_unit_interval_float(min_confidence, name="min_confidence"),
        min_margin=_unit_interval_float(min_margin, name="min_margin"),
        top_fraction=None if top_fraction in {None, "", "none", "None"} else _open_unit_float(top_fraction, name="top_fraction"),
        min_keep_rows=_nonnegative_int(min_keep_rows, name="min_keep_rows"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def _coerce_config(config: TargetConfidenceSelectionConfig | Mapping[str, Any]) -> TargetConfidenceSelectionConfig:
    if isinstance(config, TargetConfidenceSelectionConfig):
        return config
    return target_confidence_selection_config(**dict(config))


def _probability_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, epsilon: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty two-dimensional matrix with at least two columns.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probabilities must be finite and non-negative.")
    matrix = np.maximum(matrix, float(epsilon))
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must have positive mass.")
    return matrix / row_sums


def _classes(values: Sequence[Any] | np.ndarray | None, *, n_classes: int) -> np.ndarray:
    if values is None:
        return np.arange(n_classes, dtype=int)
    vector = np.asarray(values, dtype=object).reshape(-1)
    if vector.shape[0] != n_classes:
        raise ValueError(f"classes must contain one value per probability column: {vector.shape[0]} != {n_classes}.")
    if len(set(vector.tolist())) != n_classes:
        raise ValueError("classes must be unique.")
    return vector


def _top_fraction_mask(confidence: np.ndarray, *, top_fraction: float) -> np.ndarray:
    n_keep = max(1, int(np.ceil(confidence.shape[0] * top_fraction)))
    order = np.lexsort((np.arange(confidence.shape[0]), -confidence))
    mask = np.zeros(confidence.shape[0], dtype=bool)
    mask[order[:n_keep]] = True
    return mask


def _force_min_keep(confidence: np.ndarray, *, keep_mask: np.ndarray, min_keep_rows: int) -> np.ndarray:
    n_keep = min(int(min_keep_rows), confidence.shape[0])
    forced = keep_mask.copy()
    if int(np.count_nonzero(forced)) >= n_keep:
        return forced
    order = np.lexsort((np.arange(confidence.shape[0]), -confidence))
    forced[order[:n_keep]] = True
    return forced


def _unit_interval_float(value: float | str, *, name: str) -> float:
    parsed = _finite_float(value, name=name)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed


def _open_unit_float(value: float | str, *, name: str) -> float:
    parsed = _finite_float(value, name=name)
    if parsed <= 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in (0, 1].")
    return parsed


def _nonnegative_int(value: int | str, *, name: str) -> int:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(parsed)


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = _finite_float(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _finite_float(value: float | str, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed
