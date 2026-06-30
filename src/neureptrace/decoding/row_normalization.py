"""Row-wise feature normalization helpers.

This module applies deterministic per-row normalization to source and held-out
feature matrices.  No cross-row statistics are fitted from held-out data; each row
is transformed independently, making it a lightweight Protocol-1 preprocessing
helper for decoders that benefit from unit-norm feature vectors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ROW_NORMALIZATION_PROTOCOL = "row_wise_feature_normalization"
ROW_NORMALIZATION_CATEGORY = "1_strict_source_only"
NORM_MODES = ("l2", "l1", "max")
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class RowNormalizationConfig:
    """Configuration for row-wise feature normalization."""

    norm: str = "l2"
    center_rows: bool = False
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class RowNormalizationResult:
    """Normalized source/test matrices and protocol metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    train_norms: np.ndarray
    test_norms: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments
def normalize_source_and_test_rows(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: RowNormalizationConfig | Mapping[str, Any] | None = None,
) -> RowNormalizationResult:
    """Normalize source and test rows independently.

    The transformation is deterministic per row and does not estimate a target
    batch statistic.  Test rows are transformed only to make them comparable with
    normalized source rows.
    """

    cfg = row_normalization_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_features and test_features must have the same feature width: {source.shape[1]} != {test.shape[1]}.")
    train, train_norms = normalize_rows(source, config=cfg)
    test_out, test_norms = normalize_rows(test, config=cfg)
    metadata = {
        "row_normalization": True,
        "row_normalization_protocol": ROW_NORMALIZATION_PROTOCOL,
        "row_normalization_protocol_category": ROW_NORMALIZATION_CATEGORY,
        "row_normalization_norm": cfg.norm,
        "row_normalization_center_rows": bool(cfg.center_rows),
        "row_normalization_uses_cross_row_source_statistics": False,
        "row_normalization_uses_cross_row_test_statistics": False,
        "row_normalization_uses_test_labels": False,
        "row_normalization_valid_for_strict_source_only": True,
        "row_normalization_valid_for_benchmark": True,
        "row_normalization_n_source_rows": int(source.shape[0]),
        "row_normalization_n_test_rows": int(test.shape[0]),
        "row_normalization_feature_dim": int(source.shape[1]),
        "row_normalization_epsilon": float(cfg.epsilon),
    }
    return RowNormalizationResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        train_norms=train_norms.astype(np.float32, copy=False),
        test_norms=test_norms.astype(np.float32, copy=False),
        metadata=metadata,
    )


def normalize_rows(
    features: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: RowNormalizationConfig | Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each feature row independently and return original row norms."""

    cfg = row_normalization_config() if config is None else _coerce_config(config)
    matrix = _feature_matrix(features, name="features")
    working = matrix - np.mean(matrix, axis=1, keepdims=True) if cfg.center_rows else matrix.copy()
    norms = row_norms(working, norm=cfg.norm)
    safe_norms = np.maximum(norms, cfg.epsilon)
    return working / safe_norms[:, None], norms


def row_norms(features: Sequence[Sequence[float]] | np.ndarray, *, norm: str = "l2") -> np.ndarray:
    """Return one norm value per row."""

    matrix = _feature_matrix(features, name="features")
    mode = normalize_norm_mode(norm)
    if mode == "l2":
        return np.sqrt(np.sum(matrix * matrix, axis=1))
    if mode == "l1":
        return np.sum(np.abs(matrix), axis=1)
    if mode == "max":
        return np.max(np.abs(matrix), axis=1)
    raise ValueError(f"Unhandled row norm mode {mode!r}.")


def row_normalization_config(
    *,
    norm: str | None = "l2",
    center_rows: bool | int | str = False,
    epsilon: float | str = DEFAULT_EPSILON,
) -> RowNormalizationConfig:
    """Normalize row-normalization options."""

    return RowNormalizationConfig(
        norm=normalize_norm_mode(norm),
        center_rows=_bool_value(center_rows, name="center_rows"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def normalize_norm_mode(value: str | None) -> str:
    """Normalize norm aliases."""

    normalized = "l2" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"euclidean": "l2", "manhattan": "l1", "linf": "max", "inf": "max"}.get(normalized, normalized)
    if normalized not in NORM_MODES:
        raise ValueError(f"Unknown row norm mode {value!r}.")
    return normalized


def _coerce_config(config: RowNormalizationConfig | Mapping[str, Any]) -> RowNormalizationConfig:
    if isinstance(config, RowNormalizationConfig):
        return config
    return row_normalization_config(**dict(config))


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _bool_value(value: bool | int | str, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean value.")
