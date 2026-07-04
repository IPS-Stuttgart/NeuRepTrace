"""Source-compatible row normalization transforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ROW_NORMALIZATION_PROTOCOL = "strict_source_compatible_row_normalization"
ROW_NORMALIZATION_CATEGORY = "1_strict_source_only"
ROW_NORMALIZATION_MODES = ("none", "l2", "l1", "max_abs")
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class RowNormalizationConfig:
    """Configuration for row-wise feature normalization."""

    mode: str = "l2"
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class RowNormalizationResult:
    """Normalized train/eval matrices and metadata."""

    train_features: np.ndarray
    eval_features: np.ndarray
    train_norms: np.ndarray
    eval_norms: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def apply_row_normalization_pair(
    *,
    train_features: Sequence[Sequence[float]] | np.ndarray,
    eval_features: Sequence[Sequence[float]] | np.ndarray,
    config: RowNormalizationConfig | Mapping[str, Any] | None = None,
) -> RowNormalizationResult:
    """Apply one row-wise normalization rule to train and eval rows."""

    cfg = row_normalization_config() if config is None else _coerce_config(config)
    train = _feature_matrix(train_features, name="train_features")
    eval_matrix = _feature_matrix(eval_features, name="eval_features")
    if train.shape[1] != eval_matrix.shape[1]:
        raise ValueError(f"train_features and eval_features must have the same feature width: {train.shape[1]} != {eval_matrix.shape[1]}.")
    train_normalized, train_norms = normalize_feature_rows(train, mode=cfg.mode, epsilon=cfg.epsilon)
    eval_normalized, eval_norms = normalize_feature_rows(eval_matrix, mode=cfg.mode, epsilon=cfg.epsilon)
    return RowNormalizationResult(
        train_features=train_normalized.astype(np.float32, copy=False),
        eval_features=eval_normalized.astype(np.float32, copy=False),
        train_norms=train_norms.astype(np.float32, copy=False),
        eval_norms=eval_norms.astype(np.float32, copy=False),
        metadata=_metadata(cfg, n_train_rows=train.shape[0], n_eval_rows=eval_matrix.shape[0], feature_dim=train.shape[1]),
    )


def row_normalization_config(*, mode: str | None = "l2", epsilon: float | str = DEFAULT_EPSILON) -> RowNormalizationConfig:
    """Normalize public row-normalization options."""

    return RowNormalizationConfig(mode=normalize_row_normalization_mode(mode), epsilon=_positive_float(epsilon, name="epsilon"))


def normalize_row_normalization_mode(value: str | None) -> str:
    """Normalize row-normalization mode aliases."""

    normalized = "l2" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"off": "none", "identity": "none", "euclidean": "l2", "manhattan": "l1", "linf": "max_abs", "max": "max_abs"}.get(normalized, normalized)
    if normalized not in ROW_NORMALIZATION_MODES:
        raise ValueError(f"Unknown row normalization mode {value!r}. Available values: {', '.join(ROW_NORMALIZATION_MODES)}.")
    return normalized


def normalize_feature_rows(features: Sequence[Sequence[float]] | np.ndarray, *, mode: str = "l2", epsilon: float = DEFAULT_EPSILON) -> tuple[np.ndarray, np.ndarray]:
    """Normalize rows and return normalized rows plus raw row norms."""

    matrix = _feature_matrix(features, name="features")
    resolved = normalize_row_normalization_mode(mode)
    eps = _positive_float(epsilon, name="epsilon")
    if resolved == "none":
        return matrix.astype(np.float32, copy=False), np.ones(matrix.shape[0], dtype=np.float32)
    if resolved == "l2":
        norms = np.linalg.norm(matrix, ord=2, axis=1)
    elif resolved == "l1":
        norms = np.linalg.norm(matrix, ord=1, axis=1)
    elif resolved == "max_abs":
        norms = np.max(np.abs(matrix), axis=1)
    else:
        raise ValueError(f"Unhandled row normalization mode {resolved!r}.")
    safe_norms = np.maximum(norms, eps)
    return (matrix / safe_norms[:, None]).astype(np.float32, copy=False), norms.astype(np.float32, copy=False)


def _coerce_config(config: RowNormalizationConfig | Mapping[str, Any]) -> RowNormalizationConfig:
    if isinstance(config, RowNormalizationConfig):
        return config
    return row_normalization_config(**dict(config))


def _metadata(cfg: RowNormalizationConfig, *, n_train_rows: int, n_eval_rows: int, feature_dim: int) -> dict[str, Any]:
    return {
        "row_normalization": cfg.mode != "none",
        "row_normalization_protocol": ROW_NORMALIZATION_PROTOCOL,
        "row_normalization_protocol_category": ROW_NORMALIZATION_CATEGORY,
        "row_normalization_mode": cfg.mode,
        "row_normalization_uses_train_features": True,
        "row_normalization_fits_eval_statistics": False,
        "row_normalization_uses_eval_labels": False,
        "row_normalization_valid_for_strict_source_only": True,
        "row_normalization_valid_for_benchmark": True,
        "row_normalization_n_train_rows": int(n_train_rows),
        "row_normalization_n_eval_rows": int(n_eval_rows),
        "row_normalization_feature_dim": int(feature_dim),
        "row_normalization_epsilon": float(cfg.epsilon),
    }


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
