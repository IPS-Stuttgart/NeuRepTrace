"""Strict source-only univariate feature selection.

The helpers in this module score feature columns from source rows and source
labels only, then apply the fixed selected-column mask to source and held-out
feature matrices.  This is a Protocol-1 preprocessing baseline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_FEATURE_SELECTION_PROTOCOL = "strict_source_only_univariate_feature_selection"
SOURCE_FEATURE_SELECTION_CATEGORY = "1_strict_source_only"
SCORE_METHODS = ("anova", "variance")
DEFAULT_K = 128
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class SourceFeatureSelectionConfig:
    """Configuration for source-only feature selection."""

    method: str = "anova"
    k: int | str = DEFAULT_K
    min_score: float | None = None
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class SourceFeatureSelectionResult:
    """Selected source/test features and provenance metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    selected_indices: np.ndarray
    scores: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments
def fit_source_feature_selection(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceFeatureSelectionConfig | Mapping[str, Any] | None = None,
) -> SourceFeatureSelectionResult:
    """Fit feature selection on source rows and transform source/test rows."""

    cfg = source_feature_selection_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_features and test_features must have the same feature width: {source.shape[1]} != {test.shape[1]}.")
    labels = np.asarray(source_labels, dtype=object).reshape(-1)
    if labels.shape[0] != source.shape[0]:
        raise ValueError(f"source_labels must contain one value per source row: {labels.shape[0]} != {source.shape[0]}.")
    scores = source_feature_scores(source, labels, method=cfg.method, epsilon=cfg.epsilon)
    selected = select_top_source_features(scores, k=cfg.k, min_score=cfg.min_score)
    metadata = _metadata(cfg, n_source_rows=source.shape[0], n_test_rows=test.shape[0], feature_dim=source.shape[1], n_selected=selected.shape[0])
    return SourceFeatureSelectionResult(
        train_features=source[:, selected].astype(np.float32, copy=False),
        test_features=test[:, selected].astype(np.float32, copy=False),
        selected_indices=selected,
        scores=scores.astype(np.float32, copy=False),
        metadata=metadata,
    )


def source_feature_scores(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    *,
    method: str = "anova",
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Score features using source rows only."""

    source = _feature_matrix(source_features, name="source_features")
    labels = np.asarray(source_labels, dtype=object).reshape(-1)
    if labels.shape[0] != source.shape[0]:
        raise ValueError("source_labels must contain one value per source row.")
    resolved = normalize_score_method(method)
    if resolved == "variance":
        return np.var(source, axis=0, ddof=1 if source.shape[0] > 1 else 0)
    epsilon_value = _positive_float(epsilon, name="epsilon")
    classes = tuple(dict.fromkeys(labels.tolist()))
    if len(classes) < 2:
        raise ValueError("ANOVA feature scoring requires at least two classes.")
    overall = np.mean(source, axis=0)
    between = np.zeros(source.shape[1], dtype=float)
    within = np.zeros(source.shape[1], dtype=float)
    for label in classes:
        mask = labels == label
        rows = source[mask]
        if rows.shape[0] == 0:
            continue
        mean = np.mean(rows, axis=0)
        between += rows.shape[0] * (mean - overall) ** 2
        within += np.sum((rows - mean) ** 2, axis=0)
    df_between = max(len(classes) - 1, 1)
    df_within = max(source.shape[0] - len(classes), 1)
    return (between / df_between) / np.maximum(within / df_within, epsilon_value)


def select_top_source_features(scores: Sequence[float] | np.ndarray, *, k: int | str = DEFAULT_K, min_score: float | str | None = None) -> np.ndarray:
    """Return sorted selected feature indices from source-only scores."""

    vector = np.asarray(scores, dtype=float).reshape(-1)
    if vector.shape[0] < 1 or not np.all(np.isfinite(vector)):
        raise ValueError("scores must be a non-empty finite vector.")
    limit = _resolve_k(k, n_features=vector.shape[0])
    order = np.argsort(-vector, kind="mergesort")[:limit]
    if not _is_missing_min_score(min_score):
        threshold = _finite_float(min_score, name="min_score")
        order = order[vector[order] >= threshold]
    if order.size == 0:
        raise ValueError("No features selected; relax k or min_score.")
    return np.sort(order.astype(int, copy=False))


def source_feature_selection_config(
    *,
    method: str | None = "anova",
    k: int | str = DEFAULT_K,
    min_score: float | str | None = None,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceFeatureSelectionConfig:
    """Normalize feature-selection options."""

    return SourceFeatureSelectionConfig(
        method=normalize_score_method(method),
        k=_validate_k_option(k),
        min_score=None if _is_missing_min_score(min_score) else _finite_float(min_score, name="min_score"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def normalize_score_method(value: str | None) -> str:
    """Normalize feature-score method aliases."""

    normalized = "anova" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"f": "anova", "f_score": "anova", "fscore": "anova", "var": "variance"}.get(normalized, normalized)
    if normalized not in SCORE_METHODS:
        raise ValueError(f"Unknown source feature score method {value!r}.")
    return normalized


def _coerce_config(config: SourceFeatureSelectionConfig | Mapping[str, Any]) -> SourceFeatureSelectionConfig:
    if isinstance(config, SourceFeatureSelectionConfig):
        return source_feature_selection_config(method=config.method, k=config.k, min_score=config.min_score, epsilon=config.epsilon)
    return source_feature_selection_config(**dict(config))


def _metadata(cfg: SourceFeatureSelectionConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int, n_selected: int) -> dict[str, Any]:
    return {
        "source_feature_selection": True,
        "source_feature_selection_protocol": SOURCE_FEATURE_SELECTION_PROTOCOL,
        "source_feature_selection_protocol_category": SOURCE_FEATURE_SELECTION_CATEGORY,
        "source_feature_selection_method": cfg.method,
        "source_feature_selection_uses_source_features": True,
        "source_feature_selection_uses_source_labels": cfg.method == "anova",
        "source_feature_selection_uses_test_features_for_fitting": False,
        "source_feature_selection_uses_test_labels": False,
        "source_feature_selection_valid_for_strict_source_only": True,
        "source_feature_selection_valid_for_benchmark": True,
        "source_feature_selection_n_source_rows": int(n_source_rows),
        "source_feature_selection_n_test_rows": int(n_test_rows),
        "source_feature_selection_feature_dim": int(feature_dim),
        "source_feature_selection_n_selected": int(n_selected),
        "source_feature_selection_k": str(cfg.k),
        "source_feature_selection_min_score": "" if cfg.min_score is None else float(cfg.min_score),
    }


def _resolve_k(value: int | str, *, n_features: int) -> int:
    normalized = _validate_k_option(value)
    if isinstance(normalized, str):
        return int(n_features)
    return min(int(normalized), int(n_features))


def _validate_k_option(value: int | str) -> int | str:
    message = "k must be a positive integer, 'all', or 'full'."
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"all", "full"}:
            return text
        numeric = _numeric_scalar(text, message=message)
    else:
        numeric = _numeric_scalar(value, message=message)
    if numeric % 1.0 != 0.0 or numeric < 1:
        raise ValueError(message)
    return int(numeric)


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _is_missing_min_score(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none"}
    return False


def _finite_float(value: float | str, *, name: str) -> float:
    return _numeric_scalar(value, message=f"{name} must be a finite numeric scalar.")


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = _numeric_scalar(value, message=f"{name} must be positive and finite.")
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _numeric_scalar(value: Any, *, message: str) -> float:
    if isinstance(value, (bool, np.bool_)) or isinstance(value, np.ndarray):
        raise ValueError(message)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if not np.isfinite(parsed):
        raise ValueError(message)
    return parsed
