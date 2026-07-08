"""Fold-safe row L1 normalization.

This module applies deterministic per-row L1 normalization to train and test
feature matrices.  It has no fitted parameters and does not inspect labels, so it
is safe to compose with strict source-only decoders.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ROW_L1_PROTOCOL = "fixed_row_l1_normalization"
ROW_L1_CATEGORY = "1_strict_source_only_compatible"
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class RowL1Config:
    """Configuration for row L1 normalization."""

    epsilon: float = DEFAULT_EPSILON

    def __post_init__(self) -> None:
        object.__setattr__(self, "epsilon", _positive_float(self.epsilon, name="epsilon"))


@dataclass(frozen=True, slots=True)
class RowL1Result:
    """Normalized matrices and provenance metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    train_norms: np.ndarray
    test_norms: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_train_test_rows_l1(
    *,
    train_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: RowL1Config | Mapping[str, Any] | None = None,
) -> RowL1Result:
    """Apply per-row L1 normalization to train and test matrices."""

    cfg = row_l1_config() if config is None else _coerce_config(config)
    train = _feature_matrix(train_features, name="train_features")
    test = _feature_matrix(test_features, name="test_features")
    if train.shape[1] != test.shape[1]:
        raise ValueError(
            "train_features and test_features must have the same feature width: "
            f"{train.shape[1]} != {test.shape[1]}."
        )
    train_out, train_norms = normalize_rows_l1(train, epsilon=cfg.epsilon)
    test_out, test_norms = normalize_rows_l1(test, epsilon=cfg.epsilon)
    return RowL1Result(
        train_features=train_out.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        train_norms=train_norms.astype(np.float32, copy=False),
        test_norms=test_norms.astype(np.float32, copy=False),
        metadata={
            "row_l1_normalization": True,
            "row_l1_protocol": ROW_L1_PROTOCOL,
            "row_l1_protocol_category": ROW_L1_CATEGORY,
            "row_l1_has_fitted_parameters": False,
            "row_l1_uses_labels": False,
            "row_l1_valid_for_strict_source_only": True,
            "row_l1_valid_for_benchmark": True,
            "row_l1_n_train_rows": int(train.shape[0]),
            "row_l1_n_test_rows": int(test.shape[0]),
            "row_l1_feature_dim": int(train.shape[1]),
            "row_l1_epsilon": float(cfg.epsilon),
        },
    )


def row_l1_config(*, epsilon: float | str = DEFAULT_EPSILON) -> RowL1Config:
    """Normalize row-L1 options."""

    return RowL1Config(epsilon=_positive_float(epsilon, name="epsilon"))


def normalize_rows_l1(
    features: Sequence[Sequence[float]] | np.ndarray,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each row by its L1 norm and return original norms."""

    matrix = _feature_matrix(features, name="features")
    norms = np.sum(np.abs(matrix), axis=1)
    safe_norms = np.maximum(norms, _positive_float(epsilon, name="epsilon"))
    return matrix / safe_norms[:, None], norms


def _coerce_config(config: RowL1Config | Mapping[str, Any]) -> RowL1Config:
    if isinstance(config, RowL1Config):
        return config
    try:
        options = dict(config)
    except (TypeError, ValueError) as exc:
        raise ValueError("Row L1 config must be a mapping or RowL1Config.") from exc
    unknown = sorted(str(key) for key in options if key != "epsilon")
    if unknown:
        raise ValueError(f"Unknown row L1 config option(s): {', '.join(unknown)}.")
    return row_l1_config(**options)


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _is_boolean_scalar(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray) and value.shape == ():
        return isinstance(value.item(), (bool, np.bool_))
    return False


def _positive_float(value: float | str, *, name: str) -> float:
    if _is_boolean_scalar(value):
        raise ValueError(f"{name} must be positive and finite.")
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be positive and finite.")
        value = value.item()
        if _is_boolean_scalar(value):
            raise ValueError(f"{name} must be positive and finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed
