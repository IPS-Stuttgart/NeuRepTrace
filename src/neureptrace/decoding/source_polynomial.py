"""Strict source-only polynomial feature expansion.

This module builds a deterministic polynomial feature map from the source feature
width only, then applies the same fixed map to source and held-out rows.  It is a
Protocol-1 preprocessing helper: held-out rows are transformed but never used for
fitting, feature selection, or tuning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_POLYNOMIAL_PROTOCOL = "strict_source_only_polynomial_features"
SOURCE_POLYNOMIAL_CATEGORY = "1_strict_source_only"
DEFAULT_MAX_INTERACTIONS = "all"


@dataclass(frozen=True, slots=True)
class SourcePolynomialConfig:
    """Configuration for a deterministic source-only polynomial feature map."""

    include_bias: bool = False
    include_original: bool = True
    include_squares: bool = True
    include_interactions: bool = True
    max_interactions: int | str = DEFAULT_MAX_INTERACTIONS


@dataclass(frozen=True, slots=True)
class SourcePolynomialReference:
    """A fitted polynomial expansion reference."""

    n_input_features: int
    square_indices: np.ndarray
    interaction_pairs: tuple[tuple[int, int], ...]
    config: SourcePolynomialConfig
    output_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourcePolynomialResult:
    """Expanded source/test features and protocol metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    reference: SourcePolynomialReference
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments

def fit_source_polynomial_transform(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourcePolynomialConfig | Mapping[str, Any] | None = None,
) -> SourcePolynomialResult:
    """Build a source-width polynomial map and transform source/test rows."""

    cfg = source_polynomial_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_features and test_features must have the same feature width: {source.shape[1]} != {test.shape[1]}.")
    reference = fit_source_polynomial_reference(source.shape[1], config=cfg)
    train = apply_source_polynomial_transform(source, reference)
    test_out = apply_source_polynomial_transform(test, reference)
    metadata = _metadata(cfg, n_source_rows=source.shape[0], n_test_rows=test.shape[0], feature_dim=source.shape[1], output_dim=train.shape[1], n_interactions=len(reference.interaction_pairs))
    return SourcePolynomialResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        reference=reference,
        metadata=metadata,
    )


def fit_source_polynomial_reference(
    n_features: int | str,
    *,
    config: SourcePolynomialConfig | Mapping[str, Any] | None = None,
) -> SourcePolynomialReference:
    """Create a deterministic polynomial map for a source feature width."""

    cfg = source_polynomial_config() if config is None else _coerce_config(config)
    width = _positive_int(n_features, name="n_features")
    square_indices = np.arange(width, dtype=int) if cfg.include_squares else np.empty(0, dtype=int)
    pairs = _interaction_pairs(width, max_interactions=cfg.max_interactions) if cfg.include_interactions else ()
    output_names = _output_names(width, cfg=cfg, square_indices=square_indices, interaction_pairs=pairs)
    if not output_names:
        raise ValueError("At least one polynomial output feature must be enabled.")
    return SourcePolynomialReference(n_input_features=width, square_indices=square_indices, interaction_pairs=pairs, config=cfg, output_names=output_names)


def apply_source_polynomial_transform(features: Sequence[Sequence[float]] | np.ndarray, reference: SourcePolynomialReference) -> np.ndarray:
    """Apply a fitted polynomial feature map."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != reference.n_input_features:
        raise ValueError(f"features width {matrix.shape[1]} does not match polynomial reference width {reference.n_input_features}.")
    blocks: list[np.ndarray] = []
    if reference.config.include_bias:
        blocks.append(np.ones((matrix.shape[0], 1), dtype=float))
    if reference.config.include_original:
        blocks.append(matrix)
    if reference.square_indices.size:
        blocks.append(matrix[:, reference.square_indices] ** 2)
    if reference.interaction_pairs:
        interactions = np.column_stack([matrix[:, left] * matrix[:, right] for left, right in reference.interaction_pairs])
        blocks.append(interactions)
    if not blocks:
        raise ValueError("Polynomial reference does not contain any output blocks.")
    return np.hstack(blocks).astype(np.float32, copy=False)


def source_polynomial_config(
    *,
    include_bias: bool | int | str = False,
    include_original: bool | int | str = True,
    include_squares: bool | int | str = True,
    include_interactions: bool | int | str = True,
    max_interactions: int | str = DEFAULT_MAX_INTERACTIONS,
) -> SourcePolynomialConfig:
    """Normalize polynomial feature-map options."""

    return SourcePolynomialConfig(
        include_bias=_bool_value(include_bias, name="include_bias"),
        include_original=_bool_value(include_original, name="include_original"),
        include_squares=_bool_value(include_squares, name="include_squares"),
        include_interactions=_bool_value(include_interactions, name="include_interactions"),
        max_interactions=max_interactions,
    )


def _coerce_config(config: SourcePolynomialConfig | Mapping[str, Any]) -> SourcePolynomialConfig:
    if isinstance(config, SourcePolynomialConfig):
        return config
    return source_polynomial_config(**dict(config))


def _interaction_pairs(n_features: int, *, max_interactions: int | str) -> tuple[tuple[int, int], ...]:
    pairs = [(left, right) for left in range(n_features) for right in range(left + 1, n_features)]
    if isinstance(max_interactions, str):
        text = max_interactions.strip().lower()
        if text in {"all", "full"}:
            limit = len(pairs)
        else:
            limit = _nonnegative_int(text, name="max_interactions")
    else:
        limit = _nonnegative_int(max_interactions, name="max_interactions")
    return tuple(pairs[: min(limit, len(pairs))])


def _output_names(n_features: int, *, cfg: SourcePolynomialConfig, square_indices: np.ndarray, interaction_pairs: tuple[tuple[int, int], ...]) -> tuple[str, ...]:
    names: list[str] = []
    if cfg.include_bias:
        names.append("bias")
    if cfg.include_original:
        names.extend(f"x{index}" for index in range(n_features))
    names.extend(f"x{int(index)}^2" for index in square_indices)
    names.extend(f"x{left}*x{right}" for left, right in interaction_pairs)
    return tuple(names)


def _metadata(cfg: SourcePolynomialConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int, output_dim: int, n_interactions: int) -> dict[str, Any]:
    return {
        "source_polynomial_features": True,
        "source_polynomial_protocol": SOURCE_POLYNOMIAL_PROTOCOL,
        "source_polynomial_protocol_category": SOURCE_POLYNOMIAL_CATEGORY,
        "source_polynomial_uses_source_feature_width": True,
        "source_polynomial_uses_source_values": False,
        "source_polynomial_uses_source_labels": False,
        "source_polynomial_uses_test_features_for_fitting": False,
        "source_polynomial_uses_test_labels": False,
        "source_polynomial_valid_for_strict_source_only": True,
        "source_polynomial_valid_for_benchmark": True,
        "source_polynomial_n_source_rows": int(n_source_rows),
        "source_polynomial_n_test_rows": int(n_test_rows),
        "source_polynomial_input_dim": int(feature_dim),
        "source_polynomial_output_dim": int(output_dim),
        "source_polynomial_n_interactions": int(n_interactions),
        "source_polynomial_include_bias": bool(cfg.include_bias),
        "source_polynomial_include_original": bool(cfg.include_original),
        "source_polynomial_include_squares": bool(cfg.include_squares),
        "source_polynomial_include_interactions": bool(cfg.include_interactions),
        "source_polynomial_max_interactions": str(cfg.max_interactions),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _positive_int(value: int | str, *, name: str) -> int:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _nonnegative_int(value: int | str, *, name: str) -> int:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(parsed)


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
