"""Fold-safe row L2 normalization.

This module applies deterministic per-row L2 normalization to train and test
feature matrices.  It has no fitted parameters and does not inspect labels, so it
is safe to compose with strict source-only decoders.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ROW_L2_PROTOCOL = "fixed_row_l2_normalization"
ROW_L2_CATEGORY = "1_strict_source_only_compatible"
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class RowL2Config:
    """Configuration for row L2 normalization."""

    epsilon: float = DEFAULT_EPSILON

    def __post_init__(self) -> None:
        object.__setattr__(self, "epsilon", _positive_float(self.epsilon, name="epsilon"))


@dataclass(frozen=True, slots=True)
class RowL2Result:
    """Normalized matrices and provenance metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    train_norms: np.ndarray
    test_norms: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_train_test_rows_l2(
    *,
    train_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: RowL2Config | Mapping[str, Any] | None = None,
) -> RowL2Result:
    """Apply per-row L2 normalization to train and test matrices."""

    cfg = row_l2_config() if config is None else _coerce_config(config)
    train = _feature_matrix(train_features, name="train_features")
    test = _feature_matrix(test_features, name="test_features")
    if train.shape[1] != test.shape[1]:
        raise ValueError(
            "train_features and test_features must have the same feature width: "
            f"{train.shape[1]} != {test.shape[1]}."
        )
    train_out, train_norms = normalize_rows_l2(train, epsilon=cfg.epsilon)
    test_out, test_norms = normalize_rows_l2(test, epsilon=cfg.epsilon)
    return RowL2Result(
        train_features=train_out.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        train_norms=train_norms.astype(float, copy=False),
        test_norms=test_norms.astype(float, copy=False),
        metadata={
            "row_l2_normalization": True,
            "row_l2_protocol": ROW_L2_PROTOCOL,
            "row_l2_protocol_category": ROW_L2_CATEGORY,
            "row_l2_has_fitted_parameters": False,
            "row_l2_uses_labels": False,
            "row_l2_valid_for_strict_source_only": True,
            "row_l2_valid_for_benchmark": True,
            "row_l2_n_train_rows": int(train.shape[0]),
            "row_l2_n_test_rows": int(test.shape[0]),
            "row_l2_feature_dim": int(train.shape[1]),
            "row_l2_epsilon": float(cfg.epsilon),
        },
    )


def row_l2_config(*, epsilon: float | str = DEFAULT_EPSILON) -> RowL2Config:
    """Normalize row-L2 options."""

    return RowL2Config(epsilon=_positive_float(epsilon, name="epsilon"))


def normalize_rows_l2(
    features: Sequence[Sequence[float]] | np.ndarray,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each row by its L2 norm and return original norms."""

    matrix = _feature_matrix(features, name="features")
    norms = _stable_row_l2_norms(matrix)
    safe_norms = np.maximum(norms, _positive_float(epsilon, name="epsilon"))
    return matrix / safe_norms[:, None], norms


def _stable_row_l2_norms(matrix: np.ndarray) -> np.ndarray:
    """Return row L2 norms without overflowing on representable finite norms."""

    scales = np.max(np.abs(matrix), axis=1)
    norms = np.zeros(matrix.shape[0], dtype=float)
    nonzero = scales > 0.0
    if not np.any(nonzero):
        return norms
    scaled = matrix[nonzero] / scales[nonzero, None]
    scaled_norms = np.sqrt(np.sum(scaled * scaled, axis=1))
    with np.errstate(over="ignore"):
        norms[nonzero] = scales[nonzero] * scaled_norms
    return norms


def _coerce_config(config: RowL2Config | Mapping[str, Any]) -> RowL2Config:
    if isinstance(config, RowL2Config):
        return config
    if not isinstance(config, Mapping):
        raise ValueError("Row L2 config must be a mapping or RowL2Config.")
    options = dict(config)
    unknown = sorted(str(key) for key in options if key != "epsilon")
    if unknown:
        raise ValueError(f"Unknown row L2 config option(s): {', '.join(unknown)}.")
    return row_l2_config(**options)


def _materialize_one_pass_iterables(value: object) -> object:
    """Materialize nested one-pass feature iterables before NumPy consumes them."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        return _materialize_one_pass_iterables(value.tolist())
    if isinstance(value, (str, bytes, Mapping)):
        return value
    if isinstance(value, Iterable):
        return [_materialize_one_pass_iterables(item) for item in value]
    return value


def _contains_boolean_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return bool(value.size)
        if value.dtype == object:
            return any(_contains_boolean_value(item) for item in value.ravel(order="C"))
        return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.generic):
        return isinstance(value.item(), (bool, np.bool_))
    if isinstance(value, Mapping):
        return any(_contains_boolean_value(item) for item in value.values())
    if isinstance(value, Iterable):
        return any(_contains_boolean_value(item) for item in value)
    return False


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    materialized = _materialize_one_pass_iterables(values)
    if _contains_boolean_value(materialized):
        raise ValueError(f"{name} must contain numeric feature values, not boolean flags.")
    try:
        matrix = np.asarray(materialized, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.") from exc
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
