"""Strict source-only distance-based sample weighting.

This module computes per-row source sample weights from source feature distances
only.  Rows far from a source-fitted group center can be downweighted before
training an ordinary decoder.  The helper is Protocol 1: it uses source features,
source labels, and optional source-domain ids, but never held-out target rows or
held-out labels.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_DISTANCE_WEIGHT_PROTOCOL = "strict_source_only_distance_weighting"
SOURCE_DISTANCE_WEIGHT_CATEGORY = "1_strict_source_only"
DISTANCE_GROUP_MODES = ("global", "class", "domain", "class_domain")
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MIN_WEIGHT = 0.05
DEFAULT_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class SourceDistanceWeightConfig:
    """Configuration for source-only distance weighting."""

    group_mode: str = "class"
    temperature: float = DEFAULT_TEMPERATURE
    min_weight: float = DEFAULT_MIN_WEIGHT
    normalize_weights: bool = True
    robust: bool = True
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class SourceDistanceWeightResult:
    """Source distance scores, sample weights, and provenance metadata."""

    sample_weights: np.ndarray
    distance_scores: np.ndarray
    group_keys: tuple[Hashable, ...]
    group_centers: Mapping[Hashable, np.ndarray]
    group_scales: Mapping[Hashable, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-locals
def compute_source_distance_weights(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    *,
    source_domains: Sequence[Hashable] | np.ndarray | None = None,
    config: SourceDistanceWeightConfig | Mapping[str, Any] | None = None,
) -> SourceDistanceWeightResult:
    """Compute source-only sample weights from standardized group distances.

    Parameters
    ----------
    source_features:
        Source feature matrix.  Rows are trials/windows and columns are features.
    source_labels:
        One source label per feature row.  Used for class and class-domain modes.
    source_domains:
        Optional source-domain ids.  Used for domain and class-domain modes.
    config:
        Distance weighting settings.  A mapping is normalized through
        :func:`source_distance_weight_config`.
    """

    cfg = source_distance_weight_config() if config is None else _coerce_config(config)
    features = _feature_matrix(source_features, name="source_features")
    labels = _value_vector(source_labels, expected_length=features.shape[0], name="source_labels")
    domains = _domain_vector(source_domains, expected_length=features.shape[0])
    keys = _group_keys(labels, domains, mode=cfg.group_mode)
    key_array = _object_vector(keys)
    scores = np.zeros(features.shape[0], dtype=float)
    centers: dict[Hashable, np.ndarray] = {}
    scales: dict[Hashable, np.ndarray] = {}

    for key in _unique_values(keys):
        mask = _equal_mask(key_array, key)
        rows = features[mask]
        center, scale = _center_scale(rows, robust=cfg.robust, epsilon=cfg.epsilon)
        centers[key] = center
        scales[key] = scale
        standardized = (rows - center) / scale
        scores[mask] = np.sum(standardized * standardized, axis=1) / features.shape[1]

    raw_weights = np.exp(-scores / cfg.temperature)
    raw_weights = np.maximum(raw_weights, cfg.min_weight)
    if cfg.normalize_weights:
        raw_weights = raw_weights / float(np.mean(raw_weights))
    metadata = _metadata(
        cfg,
        n_source_rows=features.shape[0],
        feature_dim=features.shape[1],
        n_groups=len(centers),
        score_min=float(np.min(scores)),
        score_max=float(np.max(scores)),
        weight_min=float(np.min(raw_weights)),
        weight_max=float(np.max(raw_weights)),
    )
    return SourceDistanceWeightResult(
        sample_weights=raw_weights.astype(np.float32, copy=False),
        distance_scores=scores.astype(np.float32, copy=False),
        group_keys=tuple(keys),
        group_centers=centers,
        group_scales=scales,
        metadata=metadata,
    )


def source_distance_weight_config(
    *,
    group_mode: str | None = "class",
    temperature: float | str = DEFAULT_TEMPERATURE,
    min_weight: float | str = DEFAULT_MIN_WEIGHT,
    normalize_weights: bool | int | str = True,
    robust: bool | int | str = True,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceDistanceWeightConfig:
    """Normalize source-distance weighting options."""

    return SourceDistanceWeightConfig(
        group_mode=normalize_distance_group_mode(group_mode),
        temperature=_positive_float(temperature, name="temperature"),
        min_weight=_unit_interval_float(min_weight, name="min_weight"),
        normalize_weights=_bool_value(normalize_weights, name="normalize_weights"),
        robust=_bool_value(robust, name="robust"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def normalize_distance_group_mode(value: str | None) -> str:
    """Normalize distance-weighting group-mode aliases."""

    normalized = "class" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {
        "pooled": "global",
        "all": "global",
        "label": "class",
        "labels": "class",
        "subject": "domain",
        "source_domain": "domain",
        "class_subject": "class_domain",
        "domain_class": "class_domain",
    }.get(normalized, normalized)
    if normalized not in DISTANCE_GROUP_MODES:
        raise ValueError(f"Unknown source distance group mode {value!r}.")
    return normalized


def _group_keys(labels: np.ndarray, domains: np.ndarray, *, mode: str) -> list[Hashable]:
    if mode == "global":
        return ["all"] * labels.shape[0]
    if mode == "class":
        return labels.tolist()
    if mode == "domain":
        return domains.tolist()
    if mode == "class_domain":
        return [(label, domain) for label, domain in zip(labels.tolist(), domains.tolist(), strict=True)]
    raise ValueError(f"Unhandled source distance group mode {mode!r}.")


def _center_scale(rows: np.ndarray, *, robust: bool, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    if robust:
        center = np.median(rows, axis=0)
        q75 = np.percentile(rows, 75.0, axis=0)
        q25 = np.percentile(rows, 25.0, axis=0)
        scale = (q75 - q25) / 1.349
    else:
        center = np.mean(rows, axis=0)
        scale = np.std(rows - center, axis=0, ddof=1 if rows.shape[0] > 1 else 0)
    return center.astype(float, copy=False), np.maximum(scale, float(epsilon)).astype(float, copy=False)


def _metadata(
    cfg: SourceDistanceWeightConfig,
    *,
    n_source_rows: int,
    feature_dim: int,
    n_groups: int,
    score_min: float,
    score_max: float,
    weight_min: float,
    weight_max: float,
) -> dict[str, Any]:
    return {
        "source_distance_weighting": True,
        "source_distance_weighting_protocol": SOURCE_DISTANCE_WEIGHT_PROTOCOL,
        "source_distance_weighting_protocol_category": SOURCE_DISTANCE_WEIGHT_CATEGORY,
        "source_distance_weighting_group_mode": cfg.group_mode,
        "source_distance_weighting_uses_source_features": True,
        "source_distance_weighting_uses_source_labels": cfg.group_mode in {"class", "class_domain"},
        "source_distance_weighting_uses_source_domains": cfg.group_mode in {"domain", "class_domain"},
        "source_distance_weighting_uses_heldout_features": False,
        "source_distance_weighting_uses_heldout_labels": False,
        "source_distance_weighting_valid_for_strict_source_only": True,
        "source_distance_weighting_valid_for_benchmark": True,
        "source_distance_weighting_n_source_rows": int(n_source_rows),
        "source_distance_weighting_feature_dim": int(feature_dim),
        "source_distance_weighting_n_groups": int(n_groups),
        "source_distance_weighting_temperature": float(cfg.temperature),
        "source_distance_weighting_min_weight": float(cfg.min_weight),
        "source_distance_weighting_normalize_weights": bool(cfg.normalize_weights),
        "source_distance_weighting_robust": bool(cfg.robust),
        "source_distance_weighting_score_min": float(score_min),
        "source_distance_weighting_score_max": float(score_max),
        "source_distance_weighting_weight_min": float(weight_min),
        "source_distance_weighting_weight_max": float(weight_max),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _value_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    items = _value_items(values, expected_length=expected_length)
    if len(items) != expected_length:
        raise ValueError(f"{name} must contain one value per row: {len(items)} != {expected_length}.")
    return _object_vector(_hashable_value(value) for value in items)


def _value_items(values: Sequence[Any] | np.ndarray, *, expected_length: int) -> list[Any]:
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            return [array.item()]
        if array.ndim == 1:
            if array.shape[0] == expected_length:
                return array.tolist()
            if expected_length == 1:
                return [tuple(array.tolist())]
            return array.reshape(-1).tolist()
        rows = array.reshape(array.shape[0], -1)
        if rows.shape[1] == 1:
            return rows[:, 0].tolist()
        return [tuple(row.tolist()) for row in rows]
    if isinstance(values, (str, bytes)):
        return [values]
    try:
        return list(values)
    except TypeError:
        return [values]


def _domain_vector(values: Sequence[Hashable] | np.ndarray | None, *, expected_length: int) -> np.ndarray:
    if values is None:
        return np.full(expected_length, "source", dtype=object)
    vector = _value_vector(values, expected_length=expected_length, name="source_domains")
    for value in vector.tolist():
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"source_domains must be hashable; got {value!r}.") from exc
    return vector


def _object_vector(values: Sequence[Any]) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _hashable_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=object)
        if array.ndim == 0:
            return _hashable_value(array.item())
        return tuple(_hashable_value(item) for item in array.reshape(-1).tolist())
    if isinstance(value, list):
        return tuple(_hashable_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_hashable_value(item) for item in value)
    if isinstance(value, dict):
        return tuple((_hashable_value(key), _hashable_value(item)) for key, item in sorted(value.items(), key=_dict_item_sort_key))
    return value


def _dict_item_sort_key(item: tuple[Any, Any]) -> tuple[str, str, str]:
    key, _value = item
    return (type(key).__module__, type(key).__qualname__, repr(key))


def _unique_values(values: Sequence[Any]) -> tuple[Any, ...]:
    unique: list[Any] = []
    for value in values:
        if not any(_values_equal(value, existing) for existing in unique):
            unique.append(value)
    return tuple(unique)


def _equal_mask(values: np.ndarray, target: Any) -> np.ndarray:
    return np.asarray([_values_equal(value, target) for value in values.tolist()], dtype=bool)


def _values_equal(left: Any, right: Any) -> bool:
    if _is_nan_like_scalar(left) and _is_nan_like_scalar(right):
        return True
    if isinstance(left, np.generic):
        left = left.item()
    if isinstance(right, np.generic):
        right = right.item()
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return _array_values_equal(left, right)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return _sequence_values_equal(left, right)
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(equal, (bool, np.bool_)):
        return bool(equal)
    try:
        return bool(np.all(equal))
    except (TypeError, ValueError):
        return False


def _is_nan_like_scalar(value: Any) -> bool:
    if isinstance(value, np.generic):
        value = value.item()
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def _array_values_equal(left: Any, right: Any) -> bool:
    left_array = np.asarray(left, dtype=object)
    right_array = np.asarray(right, dtype=object)
    if left_array.shape != right_array.shape:
        return False
    return all(_values_equal(left_item, right_item) for left_item, right_item in zip(left_array.reshape(-1).tolist(), right_array.reshape(-1).tolist(), strict=True))


def _sequence_values_equal(left: Any, right: Any) -> bool:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return False
    if len(left) != len(right):
        return False
    return all(_values_equal(left_item, right_item) for left_item, right_item in zip(left, right, strict=True))


def _numeric_scalar(value: Any, *, message: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        raise ValueError(message)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc


def _positive_float(value: float | str, *, name: str) -> float:
    message = f"{name} must be positive and finite."
    parsed = _numeric_scalar(value, message=message)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(message)
    return parsed


def _unit_interval_float(value: float | str, *, name: str) -> float:
    message = f"{name} must be in [0, 1]."
    parsed = _numeric_scalar(value, message=message)
    if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(message)
    return parsed


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


def _coerce_config(config: SourceDistanceWeightConfig | Mapping[str, Any]) -> SourceDistanceWeightConfig:
    if isinstance(config, SourceDistanceWeightConfig):
        return source_distance_weight_config(
            group_mode=config.group_mode,
            temperature=config.temperature,
            min_weight=config.min_weight,
            normalize_weights=config.normalize_weights,
            robust=config.robust,
            epsilon=config.epsilon,
        )
    return source_distance_weight_config(**dict(config))
