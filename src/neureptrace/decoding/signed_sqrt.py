"""Fold-safe signed square-root feature transform.

This module applies a deterministic signed square-root compression to train and
test feature matrices.  It has no fitted parameters and does not inspect labels,
so it is safe to compose with strict source-only decoders.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    return SignedSqrtResult(
        train_features=signed_sqrt_transform(train, scale=cfg.scale).astype(np.float32, copy=False),
        test_features=signed_sqrt_transform(test, scale=cfg.scale).astype(np.float32, copy=False),
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
    return np.sign(matrix) * np.sqrt(np.abs(matrix) / resolved_scale)


def _coerce_config(config: SignedSqrtConfig | Mapping[str, Any]) -> SignedSqrtConfig:
    if isinstance(config, SignedSqrtConfig):
        return config
    return signed_sqrt_config(**dict(config))


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
