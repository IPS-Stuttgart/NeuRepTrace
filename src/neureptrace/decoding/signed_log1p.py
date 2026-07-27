"""Fold-safe signed log1p feature compression.

This module applies a deterministic signed ``log1p`` compression to train and
held-out feature matrices.  It has no fitted parameters and does not inspect
labels, so it is safe to compose with strict source-only decoders.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SIGNED_LOG1P_PROTOCOL = "fixed_signed_log1p_transform"
SIGNED_LOG1P_CATEGORY = "1_strict_source_only_compatible"
DEFAULT_SCALE = 1.0


@dataclass(frozen=True, slots=True)
class SignedLog1pConfig:
    """Configuration for signed log1p compression."""

    scale: float = DEFAULT_SCALE


@dataclass(frozen=True, slots=True)
class SignedLog1pResult:
    """Transformed train/test matrices and provenance metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def transform_train_test_signed_log1p(
    *,
    train_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SignedLog1pConfig | Mapping[str, Any] | None = None,
) -> SignedLog1pResult:
    """Apply fixed signed log1p compression to train and test matrices."""

    cfg = signed_log1p_config() if config is None else _coerce_config(config)
    train = _feature_matrix(train_features, name="train_features")
    test = _feature_matrix(test_features, name="test_features")
    if train.shape[1] != test.shape[1]:
        raise ValueError(
            "train_features and test_features must have the same feature width: "
            f"{train.shape[1]} != {test.shape[1]}."
        )
    return SignedLog1pResult(
        train_features=_compact_float32(signed_log1p_transform(train, scale=cfg.scale)),
        test_features=_compact_float32(signed_log1p_transform(test, scale=cfg.scale)),
        metadata={
            "signed_log1p_transform": True,
            "signed_log1p_protocol": SIGNED_LOG1P_PROTOCOL,
            "signed_log1p_protocol_category": SIGNED_LOG1P_CATEGORY,
            "signed_log1p_has_fitted_parameters": False,
            "signed_log1p_uses_labels": False,
            "signed_log1p_valid_for_strict_source_only": True,
            "signed_log1p_valid_for_benchmark": True,
            "signed_log1p_n_train_rows": int(train.shape[0]),
            "signed_log1p_n_test_rows": int(test.shape[0]),
            "signed_log1p_feature_dim": int(train.shape[1]),
            "signed_log1p_scale": float(cfg.scale),
        },
    )


def signed_log1p_config(*, scale: float | str = DEFAULT_SCALE) -> SignedLog1pConfig:
    """Normalize signed-log1p options."""

    return SignedLog1pConfig(scale=_positive_float(scale, name="scale"))


def signed_log1p_transform(features: Sequence[Sequence[float]] | np.ndarray, *, scale: float = DEFAULT_SCALE) -> np.ndarray:
    """Return ``sign(x) * log1p(abs(x) / scale)`` for each feature value."""

    matrix = _feature_matrix(features, name="features")
    resolved_scale = _positive_float(scale, name="scale")
    with np.errstate(divide="ignore"):
        log_ratio = np.log(np.abs(matrix)) - np.log(resolved_scale)
    magnitude = np.logaddexp(0.0, log_ratio)
    return np.sign(matrix) * magnitude


def _compact_float32(values: np.ndarray) -> np.ndarray:
    """Use float32 only when conversion preserves finite, nonzero values."""

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        compact = values.astype(np.float32, copy=False)
    if not np.all(np.isfinite(compact)):
        return values
    if np.any((values != 0.0) & (compact == 0.0)):
        return values
    return compact


def _coerce_config(config: SignedLog1pConfig | Mapping[str, Any]) -> SignedLog1pConfig:
    if isinstance(config, SignedLog1pConfig):
        return config
    return signed_log1p_config(**dict(config))


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
