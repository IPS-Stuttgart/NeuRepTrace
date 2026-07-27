"""Fold-safe signed square-root feature transform.

This module applies a deterministic signed square-root compression to train and
test feature matrices.  It has no fitted parameters and does not inspect labels,
so it is safe to compose with strict source-only decoders.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SIGNED_SQRT_PROTOCOL = "fixed_signed_sqrt_transform"
SIGNED_SQRT_CATEGORY = "1_strict_source_only_compatible"
DEFAULT_SCALE = 1.0


@dataclass(frozen=True, slots=True)
class SignedSqrtConfig:
    """Configuration for signed square-root compression."""

    scale: float = DEFAULT_SCALE


@dataclass(frozen=True, slots=True)
class SignedSqrtResult:
    """Transformed matrices and provenance metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def transform_train_test_signed_sqrt(
    *,
    train_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SignedSqrtConfig | Mapping[str, Any] | None = None,
) -> SignedSqrtResult:
    """Apply signed square-root compression to train and test matrices."""

    cfg = signed_sqrt_config() if config is None else _coerce_config(config)
    train = _feature_matrix(train_features, name="train_features")
    test = _feature_matrix(test_features, name="test_features")
    if train.shape[1] != test.shape[1]:
        raise ValueError(
            "train_features and test_features must have the same feature width: "
            f"{train.shape[1]} != {test.shape[1]}."
        )
    train_transformed = signed_sqrt_transform(train, scale=cfg.scale)
    test_transformed = signed_sqrt_transform(test, scale=cfg.scale)
    return SignedSqrtResult(
        train_features=_compact_float32(train_transformed),
        test_features=_compact_float32(test_transformed),
        metadata={
            "signed_sqrt_transform": True,
            "signed_sqrt_protocol": SIGNED_SQRT_PROTOCOL,
            "signed_sqrt_protocol_category": SIGNED_SQRT_CATEGORY,
            "signed_sqrt_has_fitted_parameters": False,
            "signed_sqrt_uses_labels": False,
            "signed_sqrt_valid_for_strict_source_only": True,
            "signed_sqrt_valid_for_benchmark": True,
            "signed_sqrt_n_train_rows": int(train.shape[0]),
            "signed_sqrt_n_test_rows": int(test.shape[0]),
            "signed_sqrt_feature_dim": int(train.shape[1]),
            "signed_sqrt_scale": float(cfg.scale),
        },
    )


def signed_sqrt_config(*, scale: float | str = DEFAULT_SCALE) -> SignedSqrtConfig:
    """Normalize signed-square-root options."""

    return SignedSqrtConfig(scale=_positive_float(scale, name="scale"))


def signed_sqrt_transform(features: Sequence[Sequence[float]] | np.ndarray, *, scale: float = DEFAULT_SCALE) -> np.ndarray:
    """Return ``sign(x) * sqrt(abs(x) / scale)`` for each feature value."""

    matrix = _feature_matrix(features, name="features")
    resolved_scale = _positive_float(scale, name="scale")
    return np.sign(matrix) * (np.sqrt(np.abs(matrix)) / np.sqrt(resolved_scale))


def _compact_float32(values: np.ndarray) -> np.ndarray:
    """Use float32 only when conversion preserves finite, nonzero values."""

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        compact = values.astype(np.float32, copy=False)
    if not np.all(np.isfinite(compact)):
        return values
    if np.any((values != 0.0) & (compact == 0.0)):
        return values
    return compact


def _coerce_config(config: SignedSqrtConfig | Mapping[str, Any]) -> SignedSqrtConfig:
    if isinstance(config, SignedSqrtConfig):
        return signed_sqrt_config(scale=config.scale)
    return signed_sqrt_config(**dict(config))


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    materialized = _materialize_one_pass_iterable(values)
    try:
        object_values = np.asarray(materialized, dtype=object)
    except (TypeError, ValueError):
        object_values = None
    if object_values is not None and any(isinstance(value, (bool, np.bool_)) for value in object_values.flat):
        raise ValueError(f"{name} must contain numeric feature values, not boolean flags.")
    if object_values is not None and any(isinstance(value, (complex, np.complexfloating)) for value in object_values.flat):
        raise ValueError(f"{name} must contain real-valued feature values, not complex values.")
    try:
        matrix = np.asarray(materialized, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-empty two-dimensional numeric matrix.") from exc
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
    if isinstance(values, Iterable):
        return [_materialize_one_pass_iterable(value) for value in values]
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
