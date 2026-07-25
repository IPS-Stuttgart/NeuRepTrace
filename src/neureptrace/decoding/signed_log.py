"""Fold-safe signed log feature transform.

This module applies a deterministic signed ``log1p`` compression to train and
held-out feature matrices.  It has no fitted parameters and does not inspect
labels, so it is safe to compose with strict source-only decoders.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SIGNED_LOG_PROTOCOL = "fixed_signed_log1p_transform"
SIGNED_LOG_CATEGORY = "1_strict_source_only_compatible"
DEFAULT_SCALE = 1.0


@dataclass(frozen=True, slots=True)
class SignedLogConfig:
    """Configuration for signed log compression."""

    scale: float = DEFAULT_SCALE


@dataclass(frozen=True, slots=True)
class SignedLogResult:
    """Transformed train/test matrices and provenance metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def transform_train_test_signed_log(
    *,
    train_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SignedLogConfig | Mapping[str, Any] | None = None,
) -> SignedLogResult:
    """Apply signed log compression to train and test matrices."""

    cfg = signed_log_config() if config is None else _coerce_config(config)
    train = _feature_matrix(train_features, name="train_features")
    test = _feature_matrix(test_features, name="test_features")
    if train.shape[1] != test.shape[1]:
        raise ValueError(
            "train_features and test_features must have the same feature width: "
            f"{train.shape[1]} != {test.shape[1]}."
        )
    return SignedLogResult(
        train_features=transform_signed_log(train, scale=cfg.scale).astype(np.float32, copy=False),
        test_features=transform_signed_log(test, scale=cfg.scale).astype(np.float32, copy=False),
        metadata={
            "signed_log_transform": True,
            "signed_log_protocol": SIGNED_LOG_PROTOCOL,
            "signed_log_protocol_category": SIGNED_LOG_CATEGORY,
            "signed_log_has_fitted_parameters": False,
            "signed_log_uses_labels": False,
            "signed_log_valid_for_strict_source_only": True,
            "signed_log_valid_for_benchmark": True,
            "signed_log_n_train_rows": int(train.shape[0]),
            "signed_log_n_test_rows": int(test.shape[0]),
            "signed_log_feature_dim": int(train.shape[1]),
            "signed_log_scale": float(cfg.scale),
        },
    )


def signed_log_config(*, scale: float | str = DEFAULT_SCALE) -> SignedLogConfig:
    """Normalize signed-log options."""

    return SignedLogConfig(scale=_positive_float(scale, name="scale"))


def transform_signed_log(features: Sequence[Sequence[float]] | np.ndarray, *, scale: float = DEFAULT_SCALE) -> np.ndarray:
    """Return ``sign(x) * log1p(abs(x) / scale)`` for each feature value."""

    matrix = _feature_matrix(features, name="features")
    scale_value = _positive_float(scale, name="scale")
    magnitude = np.abs(matrix)
    larger_than_scale = magnitude > scale_value
    compressed = np.empty_like(magnitude)
    with np.errstate(under="ignore"):
        compressed[~larger_than_scale] = np.log1p(magnitude[~larger_than_scale] / scale_value)
        compressed[larger_than_scale] = (
            np.log(magnitude[larger_than_scale])
            - np.log(scale_value)
            + np.log1p(scale_value / magnitude[larger_than_scale])
        )
    return np.sign(matrix) * compressed


def _coerce_config(config: SignedLogConfig | Mapping[str, Any]) -> SignedLogConfig:
    if isinstance(config, SignedLogConfig):
        return signed_log_config(scale=config.scale)
    return signed_log_config(**dict(config))


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(_materialize_one_pass_iterable(values), dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _materialize_one_pass_iterable(values: Any) -> Any:
    if isinstance(values, np.ndarray):
        return values
    if isinstance(values, (str, bytes, Mapping)):
        return values
    if hasattr(values, "__array__"):
        return values
    if isinstance(values, Iterable) and not isinstance(values, Sequence):
        return list(values)
    return values


def _positive_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be positive and finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed
