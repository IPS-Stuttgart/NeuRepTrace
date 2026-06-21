"""Dynamic source-subject selection and weighting utilities.

The helpers in this module are deliberately protocol-explicit:

* ``source_reliability`` converts source-only validation scores into fold-local
  source-group multipliers.  It uses only source data and source labels and is
  therefore compatible with strict source-only Protocol 1.
* ``target_similarity`` derives multipliers from unlabeled target feature
  summaries.  It uses target features but no target labels and should be
  reported as Protocol 2, unlabeled target-adaptive.
* ``hybrid`` combines source reliability with unlabeled target similarity and is
  likewise Protocol 2.

The output weights are mean-one over the candidate source groups so callers can
pass them directly as sample-weight multipliers without changing the effective
regularization scale too much.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

SOURCE_GROUP_WEIGHTING_MODES = ("none", "source_reliability", "target_similarity", "hybrid")
SOURCE_GROUP_MINIMIZE_METRICS = {"log_loss", "brier", "ece", "loss", "error", "mae", "mse", "rmse"}
DEFAULT_SOURCE_GROUP_WEIGHTING_TEMPERATURE = 0.25
DEFAULT_SOURCE_GROUP_WEIGHTING_BLEND = 1.0
DEFAULT_HYBRID_TARGET_SIMILARITY_WEIGHT = 0.50


@dataclass(frozen=True, slots=True)
class SourceGroupWeightingConfig:
    """Configuration for fold-local dynamic source-subject weighting."""

    mode: str = "none"
    metric: str = "balanced_accuracy"
    temperature: float = DEFAULT_SOURCE_GROUP_WEIGHTING_TEMPERATURE
    top_k: int | None = None
    blend: float = DEFAULT_SOURCE_GROUP_WEIGHTING_BLEND
    hybrid_target_similarity_weight: float = DEFAULT_HYBRID_TARGET_SIMILARITY_WEIGHT

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    @property
    def uses_unlabeled_target_data(self) -> bool:
        return self.mode in {"target_similarity", "hybrid"}

    @property
    def protocol(self) -> str:
        if not self.enabled:
            return "strict_source_only"
        if self.uses_unlabeled_target_data:
            return "unlabeled_target_adaptive"
        return "strict_source_only"

    def metadata(self) -> dict[str, Any]:
        """Return provenance fields for reports and CSV outputs."""

        return {
            "source_group_weighting": self.mode,
            "source_group_weighting_metric": self.metric if self.mode in {"source_reliability", "hybrid"} else "",
            "source_group_weighting_temperature": "" if not self.enabled else self.temperature,
            "source_group_weighting_top_k": "" if self.top_k is None else self.top_k,
            "source_group_weighting_blend": "" if not self.enabled else self.blend,
            "source_group_weighting_hybrid_target_similarity_weight": (
                self.hybrid_target_similarity_weight if self.mode == "hybrid" else ""
            ),
            "source_group_weighting_uses_unlabeled_target_data": self.uses_unlabeled_target_data,
            "source_group_weighting_uses_target_labels": False,
            "source_group_weighting_protocol": self.protocol,
            "source_group_weighting_protocol_note": (
                "uses unlabeled target features; report separately from strict source-only"
                if self.uses_unlabeled_target_data
                else ""
            ),
        }


def normalize_source_group_weighting_mode(value: Any) -> str:
    """Normalize source-group weighting mode aliases."""

    normalized = "none" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "": "none",
        "false": "none",
        "off": "none",
        "no": "none",
        "uniform": "none",
        "reliability": "source_reliability",
        "source_score": "source_reliability",
        "source_scores": "source_reliability",
        "inner_loso": "source_reliability",
        "source_inner_loso": "source_reliability",
        "target": "target_similarity",
        "target_features": "target_similarity",
        "target_centroid": "target_similarity",
        "target_feature_similarity": "target_similarity",
        "similarity": "target_similarity",
        "hybrid_similarity": "hybrid",
        "reliability_plus_similarity": "hybrid",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SOURCE_GROUP_WEIGHTING_MODES:
        raise ValueError(f"Unknown source-group weighting mode {value!r}; choose one of {SOURCE_GROUP_WEIGHTING_MODES}.")
    return normalized


def source_group_weighting_config(value: Mapping[str, Any] | str | bool | None = None, **overrides: Any) -> SourceGroupWeightingConfig:
    """Build a normalized :class:`SourceGroupWeightingConfig`.

    ``value`` can be a config mapping, a mode string, a boolean flag, or ``None``.
    Explicit keyword overrides are applied after the mapping/string value.
    """

    if isinstance(value, Mapping):
        raw = dict(value)
    elif isinstance(value, bool):
        raw = {"mode": "source_reliability" if value else "none"}
    elif value is None:
        raw = {}
    else:
        raw = {"mode": value}
    raw.update(overrides)
    mode = normalize_source_group_weighting_mode(raw.get("mode", raw.get("enabled")))
    return SourceGroupWeightingConfig(
        mode=mode,
        metric=str(raw.get("metric", "balanced_accuracy")).strip().lower().replace("-", "_"),
        temperature=_positive_float(raw.get("temperature", DEFAULT_SOURCE_GROUP_WEIGHTING_TEMPERATURE), name="source_group_weighting.temperature"),
        top_k=_optional_positive_int(raw.get("top_k"), name="source_group_weighting.top_k"),
        blend=_unit_interval_float(raw.get("blend", DEFAULT_SOURCE_GROUP_WEIGHTING_BLEND), name="source_group_weighting.blend"),
        hybrid_target_similarity_weight=_unit_interval_float(
            raw.get("hybrid_target_similarity_weight", raw.get("target_similarity_weight", DEFAULT_HYBRID_TARGET_SIMILARITY_WEIGHT)),
            name="source_group_weighting.hybrid_target_similarity_weight",
        ),
    )


def dynamic_source_group_weights(
    *,
    config: SourceGroupWeightingConfig | Mapping[str, Any] | str | bool | None,
    groups: Sequence[Hashable] | None = None,
    source_scores: Mapping[Hashable, float] | None = None,
    source_features: Mapping[Hashable, Sequence[Sequence[float]] | Sequence[float] | np.ndarray] | None = None,
    target_features: Sequence[Sequence[float]] | Sequence[float] | np.ndarray | None = None,
) -> dict[Hashable, float]:
    """Return mean-one source-group weights for one outer fold.

    Parameters
    ----------
    config:
        Source-group weighting configuration.
    groups:
        Candidate source groups to include.  When omitted, the keys from the
        required score/feature mapping define the group order.
    source_scores:
        Source-only validation scores, required by ``source_reliability`` and
        ``hybrid`` modes.
    source_features, target_features:
        Unlabeled feature rows or precomputed vectors, required by
        ``target_similarity`` and ``hybrid`` modes.
    """

    cfg = config if isinstance(config, SourceGroupWeightingConfig) else source_group_weighting_config(config)
    group_list = _group_list(groups, source_scores=source_scores, source_features=source_features)
    if not group_list:
        return {}
    if cfg.mode == "none":
        return {group: 1.0 for group in group_list}
    if cfg.mode == "source_reliability":
        if source_scores is None:
            raise ValueError("source_reliability weighting requires source_scores.")
        return weights_from_scores(source_scores, groups=group_list, metric=cfg.metric, temperature=cfg.temperature, top_k=cfg.top_k, blend=cfg.blend)
    if cfg.mode == "target_similarity":
        if source_features is None or target_features is None:
            raise ValueError("target_similarity weighting requires source_features and unlabeled target_features.")
        scores = target_similarity_scores(source_features, target_features, groups=group_list)
        return weights_from_scores(scores, groups=group_list, metric="similarity", temperature=cfg.temperature, top_k=cfg.top_k, blend=cfg.blend)
    if cfg.mode == "hybrid":
        if source_scores is None:
            raise ValueError("hybrid weighting requires source_scores.")
        if source_features is None or target_features is None:
            raise ValueError("hybrid weighting requires source_features and unlabeled target_features.")
        similarity = target_similarity_scores(source_features, target_features, groups=group_list)
        hybrid_scores = combine_source_reliability_and_similarity(
            source_scores,
            similarity,
            groups=group_list,
            metric=cfg.metric,
            target_similarity_weight=cfg.hybrid_target_similarity_weight,
        )
        return weights_from_scores(hybrid_scores, groups=group_list, metric="combined_utility", temperature=cfg.temperature, top_k=cfg.top_k, blend=cfg.blend)
    raise AssertionError(f"Unhandled source-group weighting mode: {cfg.mode}")


def weights_from_scores(
    scores: Mapping[Hashable, float],
    *,
    groups: Sequence[Hashable] | None = None,
    metric: str = "balanced_accuracy",
    temperature: float = DEFAULT_SOURCE_GROUP_WEIGHTING_TEMPERATURE,
    top_k: int | None = None,
    blend: float = DEFAULT_SOURCE_GROUP_WEIGHTING_BLEND,
) -> dict[Hashable, float]:
    """Convert validation/similarity scores to mean-one source-group weights."""

    group_list = _group_list(groups, source_scores=scores)
    if not group_list:
        return {}
    utilities = np.asarray([_score_to_utility(scores[group], metric=metric) for group in group_list], dtype=np.float64)
    utilities = _replace_nonfinite_with_minimum(utilities)
    keep = np.ones(len(group_list), dtype=bool)
    normalized_top_k = _optional_positive_int(top_k, name="source_group_weighting.top_k")
    if normalized_top_k is not None and normalized_top_k < len(group_list):
        keep[:] = False
        keep[np.argsort(utilities)[-normalized_top_k:]] = True
    raw = np.zeros(len(group_list), dtype=np.float64)
    if np.any(keep):
        kept = utilities[keep]
        temperature = _positive_float(temperature, name="source_group_weighting.temperature")
        raw[keep] = np.exp(np.clip((kept - float(np.max(kept))) / temperature, -60.0, 0.0))
    raw = _mean_one(raw)
    blend = _unit_interval_float(blend, name="source_group_weighting.blend")
    if blend < 1.0:
        raw = _mean_one((1.0 - blend) * np.ones_like(raw) + blend * raw)
    return {group: float(weight) for group, weight in zip(group_list, raw, strict=True)}


def target_similarity_scores(
    source_features: Mapping[Hashable, Sequence[Sequence[float]] | Sequence[float] | np.ndarray],
    target_features: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
    *,
    groups: Sequence[Hashable] | None = None,
) -> dict[Hashable, float]:
    """Return cosine similarity between each source feature centroid and target centroid."""

    group_list = _group_list(groups, source_features=source_features)
    target = _unit_centered_vector(_feature_centroid(target_features))
    scores: dict[Hashable, float] = {}
    for group in group_list:
        if group not in source_features:
            raise ValueError(f"Missing source features for group {group!r}.")
        source = _unit_centered_vector(_feature_centroid(source_features[group]))
        scores[group] = float(np.clip(np.dot(source, target), -1.0, 1.0))
    return scores


def combine_source_reliability_and_similarity(
    source_scores: Mapping[Hashable, float],
    target_similarity: Mapping[Hashable, float],
    *,
    groups: Sequence[Hashable] | None = None,
    metric: str = "balanced_accuracy",
    target_similarity_weight: float = DEFAULT_HYBRID_TARGET_SIMILARITY_WEIGHT,
) -> dict[Hashable, float]:
    """Return standardized hybrid utilities from source reliability and target similarity."""

    group_list = _group_list(groups, source_scores=source_scores)
    missing_similarity = [group for group in group_list if group not in target_similarity]
    if missing_similarity:
        raise ValueError(f"Missing target-similarity scores for source groups: {missing_similarity}.")
    source_utility = np.asarray([_score_to_utility(source_scores[group], metric=metric) for group in group_list], dtype=np.float64)
    target_utility = np.asarray([float(target_similarity[group]) for group in group_list], dtype=np.float64)
    source_utility = _zscore(_replace_nonfinite_with_minimum(source_utility))
    target_utility = _zscore(_replace_nonfinite_with_minimum(target_utility))
    target_similarity_weight = _unit_interval_float(target_similarity_weight, name="source_group_weighting.hybrid_target_similarity_weight")
    combined = (1.0 - target_similarity_weight) * source_utility + target_similarity_weight * target_utility
    return {group: float(value) for group, value in zip(group_list, combined, strict=True)}


def sample_weights_from_group_weights(
    row_groups: Sequence[Hashable] | np.ndarray,
    group_weights: Mapping[Hashable, float] | None,
    *,
    default: float = 1.0,
    normalize: bool = True,
) -> np.ndarray | None:
    """Expand source-group multipliers to one sample weight per training row."""

    if group_weights is None:
        return None
    rows = np.asarray(row_groups, dtype=object).reshape(-1)
    lookup = {group: _nonnegative_float(weight, name="source_group_weight") for group, weight in group_weights.items()}
    default = _nonnegative_float(default, name="source_group_weight_default")
    weights = np.asarray([lookup.get(group, default) for group in rows], dtype=np.float64)
    if normalize:
        weights = _mean_one(weights)
    return weights


def selected_source_groups(group_weights: Mapping[Hashable, float], *, min_weight: float = 0.0) -> tuple[Hashable, ...]:
    """Return source groups whose weight is strictly above ``min_weight``."""

    min_weight = _nonnegative_float(min_weight, name="min_weight")
    return tuple(group for group, weight in group_weights.items() if float(weight) > min_weight)


def _group_list(
    groups: Sequence[Hashable] | None,
    *,
    source_scores: Mapping[Hashable, float] | None = None,
    source_features: Mapping[Hashable, Any] | None = None,
) -> list[Hashable]:
    if groups is not None:
        return list(dict.fromkeys(groups))
    if source_scores is not None:
        return list(source_scores.keys())
    if source_features is not None:
        return list(source_features.keys())
    return []


def _score_to_utility(score: Any, *, metric: str) -> float:
    value = float(score)
    metric = str(metric).strip().lower().replace("-", "_")
    return -value if metric in SOURCE_GROUP_MINIMIZE_METRICS else value


def _feature_centroid(features: Sequence[Sequence[float]] | Sequence[float] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim == 1:
        centroid = matrix
    elif matrix.ndim == 2:
        if matrix.shape[0] == 0:
            raise ValueError("Feature matrix must contain at least one row.")
        centroid = matrix.mean(axis=0)
    else:
        raise ValueError("Source/target features must be a one- or two-dimensional array.")
    if centroid.size == 0:
        raise ValueError("Feature centroid must contain at least one value.")
    return np.nan_to_num(np.asarray(centroid, dtype=np.float64).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)


def _unit_centered_vector(vector: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    vector = vector - float(vector.mean())
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > epsilon else np.zeros_like(vector)


def _replace_nonfinite_with_minimum(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if np.all(finite):
        return values
    if not np.any(finite):
        return np.zeros_like(values)
    replacement = float(np.min(values[finite]))
    return np.where(finite, values, replacement)


def _zscore(values: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = float(values.std())
    if scale <= epsilon:
        return np.zeros_like(values)
    return (values - float(values.mean())) / scale


def _mean_one(weights: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(weights < 0.0):
        raise ValueError("Source-group weights must be non-negative.")
    mean = float(weights.mean()) if weights.size else 1.0
    if mean <= epsilon:
        return np.ones_like(weights)
    return weights / mean


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return number


def _nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative.")
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
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
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer or null.")
    number = float(value)
    if not np.isfinite(number) or number % 1.0 != 0.0 or number < 1:
        raise ValueError(f"{name} must be a positive integer or null.")
    return int(number)
