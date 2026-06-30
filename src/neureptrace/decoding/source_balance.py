"""Strict source-only class/domain balancing helpers."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_BALANCE_PROTOCOL = "strict_source_only_class_domain_balancing"
SOURCE_BALANCE_CATEGORY = "1_strict_source_only"
BALANCE_STRATEGIES = ("none", "class", "domain", "class_domain")
BALANCE_TARGETS = ("max", "min", "mean")


@dataclass(frozen=True, slots=True)
class SourceBalanceConfig:
    """Configuration for source-only sample weighting and resampling."""

    strategy: str = "class_domain"
    target: str = "max"
    normalize_weights: bool = True
    random_state: int | None = 13


@dataclass(frozen=True, slots=True)
class SourceBalanceResult:
    """Per-row source weights and group metadata."""

    sample_weights: np.ndarray
    group_keys: tuple[Hashable, ...]
    group_counts: Mapping[Hashable, int]
    group_target_count: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceResampleResult:
    """Balanced source-row resample."""

    features: np.ndarray
    labels: np.ndarray
    domains: np.ndarray | None
    source_indices: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def source_balance_config(
    *,
    strategy: str | None = "class_domain",
    target: str | None = "max",
    normalize_weights: bool | str | int | float = True,
    random_state: int | str | None = 13,
) -> SourceBalanceConfig:
    """Normalize source-balancing options."""

    return SourceBalanceConfig(
        strategy=normalize_balance_strategy(strategy),
        target=normalize_balance_target(target),
        normalize_weights=_bool_config(normalize_weights, name="normalize_weights"),
        random_state=_optional_nonnegative_int(random_state, name="random_state"),
    )


def compute_source_balance_weights(
    source_labels: Sequence[Any] | np.ndarray,
    *,
    source_domains: Sequence[Hashable] | np.ndarray | None = None,
    config: SourceBalanceConfig | Mapping[str, Any] | None = None,
) -> SourceBalanceResult:
    """Compute Protocol-1 source sample weights by class/domain groups."""

    cfg = source_balance_config() if config is None else _coerce_config(config)
    labels = _vector(source_labels, name="source_labels")
    domains = _domain_vector(source_domains, expected_length=labels.shape[0])
    keys = _group_keys(labels, domains, strategy=cfg.strategy)
    counts = _count_groups(keys)
    target_count = _target_count(counts, target=cfg.target)
    if cfg.strategy == "none":
        weights = np.ones(labels.shape[0], dtype=float)
    else:
        weights = np.asarray([target_count / counts[key] for key in keys], dtype=float)
    if cfg.normalize_weights and weights.size:
        weights = weights / float(np.mean(weights))
    return SourceBalanceResult(
        sample_weights=weights.astype(np.float32, copy=False),
        group_keys=tuple(keys),
        group_counts=counts,
        group_target_count=float(target_count),
        metadata=_metadata(cfg, n_source_rows=labels.shape[0], n_groups=len(counts), group_counts=counts, group_target_count=target_count, n_output_rows=""),
    )


def resample_source_rows_balanced(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    *,
    source_domains: Sequence[Hashable] | np.ndarray | None = None,
    config: SourceBalanceConfig | Mapping[str, Any] | None = None,
) -> SourceResampleResult:
    """Resample source rows to equalize class/domain groups."""

    cfg = source_balance_config() if config is None else _coerce_config(config)
    features = _feature_matrix(source_features, name="source_features")
    labels = _vector(source_labels, name="source_labels", expected_length=features.shape[0])
    if labels.shape[0] != features.shape[0]:
        raise ValueError("source_labels must contain one value per feature row.")
    domains = _domain_vector(source_domains, expected_length=features.shape[0])
    keys = _group_keys(labels, domains, strategy=cfg.strategy)
    counts = _count_groups(keys)
    target_count = int(round(_target_count(counts, target=cfg.target)))
    rng = np.random.default_rng(cfg.random_state)
    if cfg.strategy == "none":
        indices = np.arange(features.shape[0], dtype=int)
    else:
        picked: list[int] = []
        for key in tuple(dict.fromkeys(keys)):
            group_indices = np.asarray([index for index, row_key in enumerate(keys) if row_key == key], dtype=int)
            picked.extend(rng.choice(group_indices, size=target_count, replace=group_indices.size < target_count).astype(int).tolist())
        indices = np.asarray(picked, dtype=int)
    out_domains = None if source_domains is None else domains[indices]
    metadata = _metadata(cfg, n_source_rows=features.shape[0], n_groups=len(counts), group_counts=counts, group_target_count=target_count, n_output_rows=indices.shape[0])
    metadata["source_balance_resampled"] = True
    return SourceResampleResult(features=features[indices].astype(np.float32, copy=False), labels=labels[indices], domains=out_domains, source_indices=indices, metadata=metadata)


def normalize_balance_strategy(value: str | None) -> str:
    """Normalize balance strategy aliases."""

    normalized = "class_domain" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"off": "none", "label": "class", "labels": "class", "subject": "domain", "source_domain": "domain", "class_subject": "class_domain", "domain_class": "class_domain"}.get(normalized, normalized)
    if normalized not in BALANCE_STRATEGIES:
        raise ValueError(f"Unknown balance strategy {value!r}.")
    return normalized


def normalize_balance_target(value: str | None) -> str:
    """Normalize group target aliases."""

    normalized = "max" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"largest": "max", "oversample": "max", "smallest": "min", "undersample": "min", "average": "mean"}.get(normalized, normalized)
    if normalized not in BALANCE_TARGETS:
        raise ValueError(f"Unknown balance target {value!r}.")
    return normalized


def _bool_config(value: bool | str | int | float, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off"}:
            return False
    if isinstance(value, (int, np.integer)):
        parsed = int(value)
        if parsed in {0, 1}:
            return bool(parsed)
    if isinstance(value, (float, np.floating)):
        parsed_float = float(value)
        if np.isfinite(parsed_float) and parsed_float in {0.0, 1.0}:
            return bool(parsed_float)
    raise ValueError(f"{name} must be a boolean value.")


def _is_none_random_state(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _optional_nonnegative_int(value: Any, *, name: str) -> int | None:
    if _is_none_random_state(value):
        return None
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a non-negative integer.")
        value = value.item()
    if isinstance(value, (list, tuple, dict, set)):
        raise ValueError(f"{name} must be a non-negative integer.")
    return _nonnegative_int(value, name=name)


def _coerce_config(config: SourceBalanceConfig | Mapping[str, Any]) -> SourceBalanceConfig:
    if isinstance(config, SourceBalanceConfig):
        return source_balance_config(
            strategy=config.strategy,
            target=config.target,
            normalize_weights=config.normalize_weights,
            random_state=config.random_state,
        )
    return source_balance_config(**dict(config))


def _group_keys(labels: np.ndarray, domains: np.ndarray, *, strategy: str) -> list[Hashable]:
    if strategy == "none":
        return ["all"] * labels.shape[0]
    if strategy == "class":
        return labels.tolist()
    if strategy == "domain":
        return domains.tolist()
    if strategy == "class_domain":
        return [(label, domain) for label, domain in zip(labels.tolist(), domains.tolist(), strict=True)]
    raise ValueError(f"Unhandled balance strategy {strategy!r}.")


def _count_groups(keys: Sequence[Hashable]) -> dict[Hashable, int]:
    counts: dict[Hashable, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return counts


def _target_count(counts: Mapping[Hashable, int], *, target: str) -> float:
    values = np.asarray(list(counts.values()), dtype=float)
    if target == "max":
        return float(np.max(values))
    if target == "min":
        return float(np.min(values))
    if target == "mean":
        return float(np.mean(values))
    raise ValueError(f"Unhandled balance target {target!r}.")


def _metadata(cfg: SourceBalanceConfig, *, n_source_rows: int, n_groups: int, group_counts: Mapping[Hashable, int], group_target_count: float, n_output_rows: int | str) -> dict[str, Any]:
    return {
        "source_balance": cfg.strategy != "none",
        "source_balance_protocol": SOURCE_BALANCE_PROTOCOL,
        "source_balance_protocol_category": SOURCE_BALANCE_CATEGORY,
        "source_balance_strategy": cfg.strategy,
        "source_balance_target": cfg.target,
        "source_balance_uses_source_labels": True,
        "source_balance_uses_source_domains": cfg.strategy in {"domain", "class_domain"},
        "source_balance_uses_heldout_features": False,
        "source_balance_uses_heldout_labels": False,
        "source_balance_valid_for_strict_source_only": True,
        "source_balance_valid_for_benchmark": True,
        "source_balance_n_source_rows": int(n_source_rows),
        "source_balance_n_groups": int(n_groups),
        "source_balance_group_target_count": float(group_target_count),
        "source_balance_n_output_rows": n_output_rows,
        "source_balance_normalize_weights": bool(cfg.normalize_weights),
        "source_balance_group_counts": "|".join(f"{key}:{int(count)}" for key, count in group_counts.items()),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _vector(values: Sequence[Any] | np.ndarray, *, name: str, expected_length: int | None = None) -> np.ndarray:
    vector = _atomic_value_vector(values, expected_length=expected_length, name=name)
    if vector.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one value.")
    return vector


def _domain_vector(values: Sequence[Hashable] | np.ndarray | None, *, expected_length: int) -> np.ndarray:
    if values is None:
        return np.full(expected_length, "source", dtype=object)
    return _atomic_value_vector(values, expected_length=expected_length, name="source_domains")


def _atomic_value_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int | None, name: str) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        vector = _object_value_vector([values])
    else:
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            vector = _object_value_vector([array.item()])
        elif array.ndim == 1:
            if expected_length == 1 and array.shape[0] != 1:
                vector = _object_value_vector([tuple(array.tolist())])
            else:
                vector = _object_value_vector(array.reshape(-1).tolist())
        else:
            rows = array.reshape(array.shape[0], -1)
            if rows.shape[1] == 1:
                vector = _object_value_vector(rows[:, 0].tolist())
            else:
                vector = _object_value_vector(tuple(row.tolist()) for row in rows)
    if expected_length is not None and vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one value per feature row: {vector.shape[0]} != {expected_length}.")
    return vector


def _object_value_vector(values: Sequence[Any]) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = _hashable_value(value)
    return vector


def _hashable_value(value: Any) -> Hashable:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        value = tuple(_hashable_value(item) for item in value)
    elif isinstance(value, tuple):
        value = tuple(_hashable_value(item) for item in value)
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(f"source balance grouping values must be hashable; got {value!r}.") from exc
    return value


def _nonnegative_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(parsed)
