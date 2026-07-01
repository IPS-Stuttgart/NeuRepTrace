"""Runtime patches for source-group weighting mode aliases and scalar validation."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_AFFIRMATIVE_ENABLE_ALIASES = {"1", "true", "on", "yes", "enable", "enabled"}
_DISABLED_ENABLE_ALIASES = {"0", "false", "off", "no", "disable", "disabled"}
_MODE_ALIAS_PATCH_ATTR = "_enabled_alias_patch"
_SCALAR_VALIDATION_PATCH_ATTR = "_source_weighting_scalar_validation_patch"
_FINITE_FEATURE_PATCH_ATTR = "_source_weighting_finite_feature_patch"
_CONFIG_NORMALIZATION_PATCH_ATTR = "_source_weighting_direct_config_normalization_patch"


def _normalized_alias(value: Any) -> str:
    return "none" if value is None else str(value).strip().lower().replace("-", "_")


def _numeric_scalar(value: Any, *, message: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if array.dtype.kind == "b" or array.ndim > 0:
        raise ValueError(message)
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(message)
    try:
        return float(scalar)
    except (TypeError, ValueError):
        raise ValueError(message) from None


def _feature_centroid(features: Any) -> np.ndarray:
    try:
        matrix = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("Source/target features must be numeric.") from None
    if matrix.ndim == 1:
        centroid = matrix
    elif matrix.ndim == 2:
        if matrix.shape[0] == 0:
            raise ValueError("Feature matrix must contain at least one row.")
        centroid = matrix.mean(axis=0)
    else:
        raise ValueError("Source/target features must be a one- or two-dimensional array.")
    centroid = np.asarray(centroid, dtype=np.float64).reshape(-1)
    if centroid.size == 0:
        raise ValueError("Feature centroid must contain at least one value.")
    if not np.all(np.isfinite(centroid)):
        raise ValueError("Source/target features must contain only finite values.")
    return centroid


def _score_to_utility(score: Any, *, metric: str) -> float:
    value = _numeric_scalar(score, message="source-group scores must be numeric scalars.")
    metric = str(metric).strip().lower().replace("-", "_")

    import neureptrace.decoding.source_weighting as source_weighting

    return -value if metric in source_weighting.SOURCE_GROUP_MINIMIZE_METRICS else value


def _positive_float(value: Any, *, name: str) -> float:
    message = f"{name} must be positive and finite."
    number = _numeric_scalar(value, message=message)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(message)
    return number


def _nonnegative_float(value: Any, *, name: str) -> float:
    message = f"{name} must be finite and non-negative."
    number = _numeric_scalar(value, message=message)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(message)
    return number


def _unit_interval_float(value: Any, *, name: str) -> float:
    number = _nonnegative_float(value, name=name)
    if number > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return number


def _optional_positive_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "all", "full"}:
        return None
    message = f"{name} must be a positive integer or null."
    number = _numeric_scalar(value, message=message)
    if not np.isfinite(number) or number % 1.0 != 0.0 or number < 1:
        raise ValueError(message)
    return int(number)


def selected_source_groups(group_weights: Any, *, min_weight: float = 0.0) -> tuple[Any, ...]:
    """Return source groups whose weight is strictly above ``min_weight``."""

    threshold = _nonnegative_float(min_weight, name="min_weight")
    return tuple(group for group, weight in group_weights.items() if _nonnegative_float(weight, name="source_group_weight") > threshold)


def _normalize_direct_config(config: Any) -> Any:
    import neureptrace.decoding.source_weighting as source_weighting

    if isinstance(config, source_weighting.SourceGroupWeightingConfig):
        return source_weighting.source_group_weighting_config(
            {
                "mode": config.mode,
                "metric": config.metric,
                "temperature": config.temperature,
                "top_k": config.top_k,
                "blend": config.blend,
                "hybrid_target_similarity_weight": config.hybrid_target_similarity_weight,
            }
        )
    return source_weighting.source_group_weighting_config(config)


def _make_dynamic_source_group_weights(original_dynamic_source_group_weights: Any) -> Any:
    @wraps(original_dynamic_source_group_weights)
    def dynamic_source_group_weights(
        *,
        config: Any,
        groups: Any = None,
        source_scores: Any = None,
        source_features: Any = None,
        target_features: Any = None,
    ) -> dict[Any, float]:
        return original_dynamic_source_group_weights(
            config=_normalize_direct_config(config),
            groups=groups,
            source_scores=source_scores,
            source_features=source_features,
            target_features=target_features,
        )

    return dynamic_source_group_weights


def _make_combine_source_reliability_and_similarity(source_weighting: Any, original_combine_source_reliability_and_similarity: Any) -> Any:
    @wraps(original_combine_source_reliability_and_similarity)
    def combine_source_reliability_and_similarity(
        source_scores: Any,
        target_similarity: Any,
        *,
        groups: Any = None,
        metric: str = "balanced_accuracy",
        target_similarity_weight: float = 0.50,
    ) -> dict[Any, float]:
        group_list = source_weighting._group_list(groups, source_scores=source_scores)
        try:
            present_groups = [group for group in group_list if group in target_similarity]
        except TypeError:
            present_groups = []
        for group in present_groups:
            _numeric_scalar(target_similarity[group], message="target-similarity scores must be numeric scalars.")
        return original_combine_source_reliability_and_similarity(
            source_scores,
            target_similarity,
            groups=groups,
            metric=metric,
            target_similarity_weight=target_similarity_weight,
        )

    return combine_source_reliability_and_similarity


def install() -> None:
    """Install boolean-like aliases and source-group scalar guardrails."""
    import neureptrace.decoding.source_weighting as source_weighting

    original_normalize_source_group_weighting_mode = source_weighting.normalize_source_group_weighting_mode
    if not getattr(original_normalize_source_group_weighting_mode, _MODE_ALIAS_PATCH_ATTR, False):

        @wraps(original_normalize_source_group_weighting_mode)
        def normalize_source_group_weighting_mode(value: Any) -> str:
            normalized = _normalized_alias(value)
            if normalized in _AFFIRMATIVE_ENABLE_ALIASES:
                return "source_reliability"
            if normalized in _DISABLED_ENABLE_ALIASES:
                return "none"
            return original_normalize_source_group_weighting_mode(value)

        setattr(normalize_source_group_weighting_mode, _MODE_ALIAS_PATCH_ATTR, True)
        source_weighting.normalize_source_group_weighting_mode = normalize_source_group_weighting_mode

    if not getattr(source_weighting, _FINITE_FEATURE_PATCH_ATTR, False):
        source_weighting._feature_centroid = _feature_centroid
        setattr(source_weighting, _FINITE_FEATURE_PATCH_ATTR, True)

    if not getattr(source_weighting, _SCALAR_VALIDATION_PATCH_ATTR, False):
        original_combine_source_reliability_and_similarity = source_weighting.combine_source_reliability_and_similarity
        source_weighting._score_to_utility = _score_to_utility
        source_weighting._positive_float = _positive_float
        source_weighting._nonnegative_float = _nonnegative_float
        source_weighting._unit_interval_float = _unit_interval_float
        source_weighting._optional_positive_int = _optional_positive_int
        source_weighting.selected_source_groups = selected_source_groups
        source_weighting.combine_source_reliability_and_similarity = _make_combine_source_reliability_and_similarity(
            source_weighting,
            original_combine_source_reliability_and_similarity,
        )
        setattr(source_weighting, _SCALAR_VALIDATION_PATCH_ATTR, True)

    original_dynamic_source_group_weights = source_weighting.dynamic_source_group_weights
    if not getattr(original_dynamic_source_group_weights, _CONFIG_NORMALIZATION_PATCH_ATTR, False):
        patched_dynamic_source_group_weights = _make_dynamic_source_group_weights(original_dynamic_source_group_weights)
        setattr(patched_dynamic_source_group_weights, _CONFIG_NORMALIZATION_PATCH_ATTR, True)
        source_weighting.dynamic_source_group_weights = patched_dynamic_source_group_weights


__all__ = ["install"]