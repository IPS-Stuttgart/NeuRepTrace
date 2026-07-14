"""Fold-safe row L-infinity normalization.

This module applies deterministic per-row max-absolute-value normalization to
train and test feature matrices. It has no fitted parameters and does not inspect
labels, so it is safe to compose with strict source-only decoders.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._row_normalization_validation import feature_matrix as _feature_matrix
from ._row_normalization_validation import positive_float as _positive_float

ROW_LINF_PROTOCOL = "fixed_row_linf_normalization"
ROW_LINF_CATEGORY = "1_strict_source_only_compatible"
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class RowLinfConfig:
    """Configuration for row L-infinity normalization."""

    epsilon: float = DEFAULT_EPSILON

    def __post_init__(self) -> None:
        object.__setattr__(self, "epsilon", _positive_float(self.epsilon, name="epsilon"))


@dataclass(frozen=True, slots=True)
class RowLinfResult:
    """Normalized matrices and provenance metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    train_norms: np.ndarray
    test_norms: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_train_test_rows_linf(
    *,
    train_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: RowLinfConfig | Mapping[str, Any] | None = None,
) -> RowLinfResult:
    """Apply per-row L-infinity normalization to train and test matrices."""

    cfg = row_linf_config() if config is None else _coerce_config(config)
    train = _feature_matrix(train_features, name="train_features")
    test = _feature_matrix(test_features, name="test_features")
    if train.shape[1] != test.shape[1]:
        raise ValueError(
            "train_features and test_features must have the same feature width: "
            f"{train.shape[1]} != {test.shape[1]}."
        )
    train_out, train_norms = normalize_rows_linf(train, epsilon=cfg.epsilon)
    test_out, test_norms = normalize_rows_linf(test, epsilon=cfg.epsilon)
    return RowLinfResult(
        train_features=train_out.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        train_norms=train_norms.astype(float, copy=False),
        test_norms=test_norms.astype(float, copy=False),
        metadata={
            "row_linf_normalization": True,
            "row_linf_protocol": ROW_LINF_PROTOCOL,
            "row_linf_protocol_category": ROW_LINF_CATEGORY,
            "row_linf_has_fitted_parameters": False,
            "row_linf_uses_labels": False,
            "row_linf_valid_for_strict_source_only": True,
            "row_linf_valid_for_benchmark": True,
            "row_linf_n_train_rows": int(train.shape[0]),
            "row_linf_n_test_rows": int(test.shape[0]),
            "row_linf_feature_dim": int(train.shape[1]),
            "row_linf_epsilon": float(cfg.epsilon),
        },
    )


def row_linf_config(*, epsilon: float | str = DEFAULT_EPSILON) -> RowLinfConfig:
    """Normalize row-L-infinity options."""

    return RowLinfConfig(epsilon=_positive_float(epsilon, name="epsilon"))


def normalize_rows_linf(
    features: Sequence[Sequence[float]] | np.ndarray,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each row by its max absolute value and return original norms."""

    matrix = _feature_matrix(features, name="features")
    norms = np.max(np.abs(matrix), axis=1)
    safe_norms = np.maximum(norms, _positive_float(epsilon, name="epsilon"))
    return matrix / safe_norms[:, None], norms


def _coerce_config(config: RowLinfConfig | Mapping[str, Any]) -> RowLinfConfig:
    if isinstance(config, RowLinfConfig):
        return config
    try:
        options = dict(config)
    except (TypeError, ValueError) as exc:
        raise ValueError("Row L-infinity config must be a mapping or RowLinfConfig.") from exc
    unknown = sorted(str(key) for key in options if key != "epsilon")
    if unknown:
        raise ValueError(f"Unknown row L-infinity config option(s): {', '.join(unknown)}.")
    return row_linf_config(**options)
