"""Fixed row max-absolute normalization.

This module applies a deterministic per-row max-absolute normalization to train
and score feature matrices.  It has no fitted parameters and does not inspect
labels, so it is safe to compose with strict source-only decoders.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ROW_MAXABS_PROTOCOL = "fixed_row_maxabs_normalization"
ROW_MAXABS_CATEGORY = "1_strict_source_only_compatible"
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class RowMaxAbsConfig:
    """Configuration for row max-absolute normalization."""

    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class RowMaxAbsResult:
    """Normalized train/score matrices and provenance metadata."""

    train_features: np.ndarray
    score_features: np.ndarray
    train_scales: np.ndarray
    score_scales: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_train_score_rows_maxabs(
    *,
    train_features: Sequence[Sequence[float]] | np.ndarray,
    score_features: Sequence[Sequence[float]] | np.ndarray,
    config: RowMaxAbsConfig | Mapping[str, Any] | None = None,
) -> RowMaxAbsResult:
    """Apply per-row max-absolute normalization to train and score matrices."""

    cfg = row_maxabs_config() if config is None else _coerce_config(config)
    train = _feature_matrix(train_features, name="train_features")
    score = _feature_matrix(score_features, name="score_features")
    if train.shape[1] != score.shape[1]:
        raise ValueError(
            "train_features and score_features must have the same feature width: "
            f"{train.shape[1]} != {score.shape[1]}."
        )
    train_out, train_scales = normalize_rows_maxabs(train, epsilon=cfg.epsilon)
    score_out, score_scales = normalize_rows_maxabs(score, epsilon=cfg.epsilon)
    return RowMaxAbsResult(
        train_features=train_out.astype(np.float32, copy=False),
        score_features=score_out.astype(np.float32, copy=False),
        train_scales=train_scales.astype(np.float32, copy=False),
        score_scales=score_scales.astype(np.float32, copy=False),
        metadata={
            "row_maxabs_normalization": True,
            "row_maxabs_protocol": ROW_MAXABS_PROTOCOL,
            "row_maxabs_protocol_category": ROW_MAXABS_CATEGORY,
            "row_maxabs_has_fitted_parameters": False,
            "row_maxabs_uses_labels": False,
            "row_maxabs_valid_for_strict_source_only": True,
            "row_maxabs_valid_for_benchmark": True,
            "row_maxabs_n_train_rows": int(train.shape[0]),
            "row_maxabs_n_score_rows": int(score.shape[0]),
            "row_maxabs_feature_dim": int(train.shape[1]),
            "row_maxabs_epsilon": float(cfg.epsilon),
        },
    )


def row_maxabs_config(*, epsilon: float | str = DEFAULT_EPSILON) -> RowMaxAbsConfig:
    """Normalize row max-absolute options."""

    return RowMaxAbsConfig(epsilon=_positive_float(epsilon, name="epsilon"))


def normalize_rows_maxabs(
    features: Sequence[Sequence[float]] | np.ndarray,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each row by its maximum absolute value and return row scales."""

    matrix = _feature_matrix(features, name="features")
    scales = np.max(np.abs(matrix), axis=1)
    safe = np.maximum(scales, _positive_float(epsilon, name="epsilon"))
    return matrix / safe[:, None], scales


def _coerce_config(config: RowMaxAbsConfig | Mapping[str, Any]) -> RowMaxAbsConfig:
    if isinstance(config, RowMaxAbsConfig):
        return config
    return row_maxabs_config(**dict(config))


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _positive_float(value: float | str, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed
