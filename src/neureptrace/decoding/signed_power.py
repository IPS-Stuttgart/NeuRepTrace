"""Fold-safe signed power feature transform.

This module applies a deterministic signed power compression to train and held-out
feature matrices.  It has no fitted parameters and does not inspect labels, so it
is safe to compose with strict source-only decoders.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SIGNED_POWER_PROTOCOL = "fixed_signed_power_transform"
SIGNED_POWER_CATEGORY = "1_strict_source_only_compatible"
DEFAULT_POWER = 0.5


@dataclass(frozen=True, slots=True)
class SignedPowerConfig:
    """Configuration for the signed power transform."""

    power: float = DEFAULT_POWER


@dataclass(frozen=True, slots=True)
class SignedPowerResult:
    """Transformed train/test matrices and provenance metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def transform_train_test_signed_power(
    *,
    train_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SignedPowerConfig | Mapping[str, Any] | None = None,
) -> SignedPowerResult:
    """Apply fixed signed power compression to train and test matrices."""

    cfg = signed_power_config() if config is None else _coerce_config(config)
    train = _feature_matrix(train_features, name="train_features")
    test = _feature_matrix(test_features, name="test_features")
    if train.shape[1] != test.shape[1]:
        raise ValueError(
            "train_features and test_features must have the same feature width: "
            f"{train.shape[1]} != {test.shape[1]}."
        )
    train_out = signed_power_transform(train, power=cfg.power)
    test_out = signed_power_transform(test, power=cfg.power)
    return SignedPowerResult(
        train_features=train_out.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        metadata={
            "signed_power_transform": True,
            "signed_power_protocol": SIGNED_POWER_PROTOCOL,
            "signed_power_protocol_category": SIGNED_POWER_CATEGORY,
            "signed_power_has_fitted_parameters": False,
            "signed_power_uses_labels": False,
            "signed_power_valid_for_strict_source_only": True,
            "signed_power_valid_for_benchmark": True,
            "signed_power_n_train_rows": int(train.shape[0]),
            "signed_power_n_test_rows": int(test.shape[0]),
            "signed_power_feature_dim": int(train.shape[1]),
            "signed_power_power": float(cfg.power),
        },
    )


def signed_power_config(*, power: float | str = DEFAULT_POWER) -> SignedPowerConfig:
    """Normalize signed-power options."""

    return SignedPowerConfig(power=_positive_float(power, name="power"))


def signed_power_transform(features: Sequence[Sequence[float]] | np.ndarray, *, power: float = DEFAULT_POWER) -> np.ndarray:
    """Return sign(x) * abs(x)**power for every feature value."""

    matrix = _feature_matrix(features, name="features")
    exponent = _positive_float(power, name="power")
    return np.sign(matrix) * np.power(np.abs(matrix), exponent)


def _coerce_config(config: SignedPowerConfig | Mapping[str, Any]) -> SignedPowerConfig:
    if isinstance(config, SignedPowerConfig):
        return signed_power_config(power=config.power)
    return signed_power_config(**dict(config))


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
