"""Confidence-based row selection for unlabeled adaptation batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

CONFIDENCE_SELECTION_PROTOCOL = "unlabeled_probability_confidence_selection"
CONFIDENCE_SELECTION_CATEGORY = "2_unlabeled_target_adaptive"
SELECTION_MODES = ("threshold", "top_k", "per_class_top_k")
_OPTIONAL_INT_SENTINELS = {"", "none", "null", "all", "full"}


@dataclass(frozen=True, slots=True)
class ConfidenceSelectionConfig:
    """Configuration for selecting confident probability rows."""

    mode: str = "threshold"
    threshold: float = 0.9
    top_k: int | None = None
    per_class_top_k: int | None = None
    min_margin: float = 0.0
    epsilon: float = 1e-12


@dataclass(frozen=True, slots=True)
class ConfidenceSelectionResult:
    """Selected rows and derived class ids."""

    selected_mask: np.ndarray
    predicted_indices: np.ndarray
    confidences: np.ndarray
    margins: np.ndarray
    selected_indices: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_selected(self) -> int:
        return int(np.count_nonzero(self.selected_mask))


def select_confident_probability_rows(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: ConfidenceSelectionConfig | Mapping[str, Any] | None = None,
) -> ConfidenceSelectionResult:
    """Select high-confidence rows from a probability matrix."""

    cfg = confidence_selection_config() if config is None else _coerce_config(config)
    matrix = _probability_matrix(probabilities, epsilon=cfg.epsilon)
    order = np.argsort(-matrix, axis=1, kind="mergesort")
    predicted = order[:, 0]
    second = order[:, 1] if matrix.shape[1] > 1 else order[:, 0]
    row_indices = np.arange(matrix.shape[0])
    confidences = matrix[row_indices, predicted]
    margins = confidences - matrix[row_indices, second]
    selected = margins >= cfg.min_margin
    if cfg.mode == "threshold":
        selected &= confidences >= cfg.threshold
    elif cfg.mode == "top_k":
        selected &= _top_k_mask(confidences, cfg.top_k or matrix.shape[0])
    elif cfg.mode == "per_class_top_k":
        selected &= _per_class_top_k_mask(confidences, predicted, matrix.shape[1], cfg.per_class_top_k or 1)
    else:
        raise ValueError(f"Unhandled selection mode {cfg.mode!r}.")
    return ConfidenceSelectionResult(
        selected_mask=selected,
        predicted_indices=predicted.astype(int, copy=False),
        confidences=confidences.astype(float, copy=False),
        margins=margins.astype(float, copy=False),
        selected_indices=np.flatnonzero(selected).astype(int, copy=False),
        metadata=_metadata(cfg, matrix, selected, confidences, margins),
    )


def confidence_selection_config(
    *,
    mode: str | None = "threshold",
    threshold: float | str = 0.9,
    top_k: int | str | None = None,
    per_class_top_k: int | str | None = None,
    min_margin: float | str = 0.0,
    epsilon: float | str = 1e-12,
) -> ConfidenceSelectionConfig:
    return ConfidenceSelectionConfig(
        mode=normalize_selection_mode(mode),
        threshold=_unit_interval_float(threshold, name="threshold"),
        top_k=_optional_positive_int(top_k, name="top_k"),
        per_class_top_k=_optional_positive_int(per_class_top_k, name="per_class_top_k"),
        min_margin=_unit_interval_float(min_margin, name="min_margin"),
        epsilon=_open_unit_interval_float(epsilon, name="epsilon"),
    )


def normalize_selection_mode(mode: str | None) -> str:
    normalized = "threshold" if mode is None else str(mode).strip().lower().replace("-", "_")
    normalized = {"confidence": "threshold", "topk": "top_k", "per_class": "per_class_top_k"}.get(normalized, normalized)
    if normalized not in SELECTION_MODES:
        raise ValueError(f"Unknown confidence selection mode {mode!r}. Available modes: {', '.join(SELECTION_MODES)}.")
    return normalized


def _coerce_config(config: ConfidenceSelectionConfig | Mapping[str, Any]) -> ConfidenceSelectionConfig:
    if isinstance(config, ConfidenceSelectionConfig):
        return config
    return confidence_selection_config(**dict(config))


def _top_k_mask(values: np.ndarray, k: int) -> np.ndarray:
    k = max(0, min(int(k), values.shape[0]))
    mask = np.zeros(values.shape[0], dtype=bool)
    if k:
        mask[np.argsort(-values, kind="mergesort")[:k]] = True
    return mask


def _per_class_top_k_mask(values: np.ndarray, predicted: np.ndarray, n_classes: int, k: int) -> np.ndarray:
    mask = np.zeros(values.shape[0], dtype=bool)
    for class_index in range(n_classes):
        rows = np.flatnonzero(predicted == class_index)
        if rows.size:
            keep = rows[np.argsort(-values[rows], kind="mergesort")[: min(int(k), rows.size)]]
            mask[keep] = True
    return mask


def _metadata(cfg: ConfidenceSelectionConfig, matrix: np.ndarray, selected: np.ndarray, confidences: np.ndarray, margins: np.ndarray) -> dict[str, Any]:
    return {
        "confidence_selection": True,
        "confidence_selection_protocol": CONFIDENCE_SELECTION_PROTOCOL,
        "confidence_selection_protocol_category": CONFIDENCE_SELECTION_CATEGORY,
        "confidence_selection_mode": cfg.mode,
        "confidence_selection_uses_probability_rows": True,
        "confidence_selection_uses_true_labels": False,
        "confidence_selection_valid_for_strict_source_only": False,
        "confidence_selection_valid_for_unlabeled_target_adaptation": True,
        "confidence_selection_n_rows": int(matrix.shape[0]),
        "confidence_selection_n_classes": int(matrix.shape[1]),
        "confidence_selection_n_selected": int(np.count_nonzero(selected)),
        "confidence_selection_selected_fraction": float(np.mean(selected)),
        "confidence_selection_threshold": float(cfg.threshold),
        "confidence_selection_min_margin": float(cfg.min_margin),
        "confidence_selection_mean_confidence": float(np.mean(confidences)),
        "confidence_selection_mean_selected_confidence": float(np.mean(confidences[selected])) if np.any(selected) else 0.0,
        "confidence_selection_mean_margin": float(np.mean(margins)),
    }


def _probability_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, epsilon: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("probabilities must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probabilities must contain finite non-negative values.")
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must have positive total mass.")
    matrix = np.maximum(matrix, float(epsilon))
    return matrix / np.sum(matrix, axis=1, keepdims=True)


def _optional_positive_int(value: int | str | None, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in _OPTIONAL_INT_SENTINELS:
        return None
    return _positive_int(value, name=name)


def _positive_int(value: int | str, *, name: str) -> int:
    numeric = _integer(value, name=name)
    if numeric < 1:
        raise ValueError(f"{name} must be positive.")
    return numeric


def _integer(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    return int(numeric)


def _unit_interval_float(value: float | str, *, name: str) -> float:
    numeric = _float_value(value, name=name)
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return numeric


def _open_unit_interval_float(value: float | str, *, name: str) -> float:
    numeric = _float_value(value, name=name)
    if numeric <= 0.0 or numeric >= 1.0:
        raise ValueError(f"{name} must be in (0, 1).")
    return numeric


def _positive_float(value: float | str, *, name: str) -> float:
    numeric = _float_value(value, name=name)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return numeric


def _float_value(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite.")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    return numeric
