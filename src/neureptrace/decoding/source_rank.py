"""Strict source-only empirical rank transform."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_RANK_PROTOCOL = "strict_source_only_empirical_rank_transform"
SOURCE_RANK_CATEGORY = "1_strict_source_only"
RANK_OUTPUTS = ("uniform", "centered")


@dataclass(frozen=True, slots=True)
class SourceRankReference:
    """Sorted source values used as a fixed rank reference."""

    sorted_values: np.ndarray
    output: str = "uniform"
    clip_extremes: bool = True
    epsilon: float = 1e-6


@dataclass(frozen=True, slots=True)
class SourceRankTransformResult:
    """Source/evaluation rows transformed by a source-only rank reference."""

    train_features: np.ndarray
    eval_features: np.ndarray
    reference: SourceRankReference
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_source_rank_reference(source_features, *, output: str = "uniform", clip_extremes: bool = True, epsilon: float = 1e-6) -> SourceRankReference:
    """Fit an empirical rank reference from source rows only."""

    source = _matrix(source_features, name="source_features")
    return SourceRankReference(
        sorted_values=np.sort(source, axis=0).astype(float, copy=False),
        output=normalize_rank_output(output),
        clip_extremes=bool(clip_extremes),
        epsilon=_epsilon(epsilon),
    )


def transform_source_rank_features(features, reference: SourceRankReference) -> np.ndarray:
    """Transform rows to empirical source-rank coordinates."""

    matrix = _matrix(features, name="features")
    sorted_values = _matrix(reference.sorted_values, name="sorted_values")
    if matrix.shape[1] != sorted_values.shape[1]:
        raise ValueError("features width must match source rank reference.")
    n_ref = sorted_values.shape[0]
    ranks = np.empty_like(matrix, dtype=float)
    for column in range(matrix.shape[1]):
        values = sorted_values[:, column]
        left = np.searchsorted(values, matrix[:, column], side="left")
        right = np.searchsorted(values, matrix[:, column], side="right")
        ranks[:, column] = (left + right) / (2.0 * n_ref)
    if reference.clip_extremes:
        ranks = np.clip(ranks, reference.epsilon, 1.0 - reference.epsilon)
    output = normalize_rank_output(reference.output)
    if output == "uniform":
        return ranks.astype(np.float32, copy=False)
    return (2.0 * ranks - 1.0).astype(np.float32, copy=False)


def fit_source_rank_transform(*, source_features, eval_features, output: str = "uniform", clip_extremes: bool = True, epsilon: float = 1e-6) -> SourceRankTransformResult:
    """Fit source ranks and transform source/evaluation rows."""

    source = _matrix(source_features, name="source_features")
    eval_matrix = _matrix(eval_features, name="eval_features")
    if source.shape[1] != eval_matrix.shape[1]:
        raise ValueError("source_features and eval_features must have the same feature width.")
    reference = fit_source_rank_reference(source, output=output, clip_extremes=clip_extremes, epsilon=epsilon)
    train = transform_source_rank_features(source, reference)
    eval_out = transform_source_rank_features(eval_matrix, reference)
    return SourceRankTransformResult(
        train_features=train,
        eval_features=eval_out,
        reference=reference,
        metadata={
            "source_rank_transform": True,
            "source_rank_protocol": SOURCE_RANK_PROTOCOL,
            "source_rank_protocol_category": SOURCE_RANK_CATEGORY,
            "source_rank_uses_source_features": True,
            "source_rank_uses_eval_features_for_fitting": False,
            "source_rank_uses_eval_labels": False,
            "source_rank_valid_for_strict_source_only": True,
            "source_rank_valid_for_benchmark": True,
            "source_rank_n_source_rows": int(source.shape[0]),
            "source_rank_n_eval_rows": int(eval_matrix.shape[0]),
            "source_rank_feature_dim": int(source.shape[1]),
            "source_rank_output": normalize_rank_output(output),
            "source_rank_clip_extremes": bool(clip_extremes),
            "source_rank_epsilon": float(_epsilon(epsilon)),
        },
    )


def normalize_rank_output(value: str | None) -> str:
    """Normalize output aliases."""

    normalized = "uniform" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"rank": "uniform", "percentile": "uniform", "cdf": "uniform", "signed": "centered", "minus_one_one": "centered"}.get(normalized, normalized)
    if normalized not in RANK_OUTPUTS:
        raise ValueError(f"Unknown rank output {value!r}.")
    return normalized


def _matrix(values, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _epsilon(value: float | str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0 or parsed >= 0.5:
        raise ValueError("epsilon must be finite and in (0, 0.5).")
    return parsed
