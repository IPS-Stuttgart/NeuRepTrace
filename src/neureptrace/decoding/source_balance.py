"""Strict source-only class/domain balancing helpers."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_BALANCE_PROTOCOL = "strict_source_only_class_domain_balancing"
SOURCE_BALANCE_CATEGORY = "1_strict_source_only"
SOURCE_LABEL_SMOOTHING_PROTOCOL = "strict_source_only_label_smoothing"
SOURCE_LABEL_SMOOTHING_CATEGORY = "1_strict_source_only"
BALANCE_STRATEGIES = ("none", "class", "domain", "class_domain")
BALANCE_TARGETS = ("max", "min", "mean")
LABEL_SMOOTHING_PRIORS = ("uniform", "empirical")


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


@dataclass(frozen=True, slots=True)
class SourceLabelSmoothingConfig:
    """Configuration for source-only soft-label smoothing."""

    smoothing: float = 0.1
    prior: str = "uniform"
    epsilon: float = 1e-12


@dataclass(frozen=True, slots=True)
class SourceLabelSmoothingResult:
    """Smoothed label distributions and provenance metadata."""

    labels: np.ndarray
    classes: np.ndarray
    distributions: np.ndarray
    prior_distribution: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def source_balance_config(
    *,
    strategy: str | None = "class_domain",
    target: str | None = "max",
    normalize_weights: bool = True,
    random_state: int | str | None = 13,
) -> SourceBalanceConfig:
    """Normalize source-balancing options."""

    return SourceBalanceConfig(
        strategy=normalize_balance_strategy(strategy),
        target=normalize_balance_target(target),
        normalize_weights=bool(normalize_weights),
        random_state=None if random_state in {None, "", "none", "None"} else _nonnegative_int(random_state, name="random_state"),
    )


def source_label_smoothing_config(
    *,
    smoothing: float | str = 0.1,
    prior: str | None = "uniform",
    epsilon: float | str = 1e-12,
) -> SourceLabelSmoothingConfig:
    """Normalize source-label smoothing options."""

    return SourceLabelSmoothingConfig(
        smoothing=_unit_interval_float(smoothing, name="smoothing"),
        prior=normalize_label_smoothing_prior(prior),
        epsilon=_positive_float(epsilon, name="epsilon"),
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


def smooth_source_labels(
    source_labels: Sequence[Any] | np.ndarray,
    *,
    classes: Sequence[Any] | np.ndarray | None = None,
    config: SourceLabelSmoothingConfig | Mapping[str, Any] | None = None,
) -> SourceLabelSmoothingResult:
    """Convert source labels to source-only smoothed class-probability rows.

    This helper is intended for decoders that support soft training targets.  It
    never uses held-out features or held-out labels.
    """

    cfg = source_label_smoothing_config() if config is None else _coerce_label_smoothing_config(config)
    labels = _vector(source_labels, name="source_labels")
    class_values = _classes(labels, classes)
    if class_values.shape[0] < 2:
        raise ValueError("Label smoothing requires at least two classes.")
    class_to_index = {label: index for index, label in enumerate(class_values.tolist())}
    prior_distribution = source_label_prior(labels, classes=class_values, prior=cfg.prior, epsilon=cfg.epsilon)
    distributions = np.empty((labels.shape[0], class_values.shape[0]), dtype=float)
    for row, label in enumerate(labels.tolist()):
        distributions[row] = cfg.smoothing * prior_distribution
        distributions[row, class_to_index[label]] += 1.0 - cfg.smoothing
    distributions = _normalize_probability_rows(distributions, epsilon=cfg.epsilon)
    return SourceLabelSmoothingResult(
        labels=labels,
        classes=class_values,
        distributions=distributions.astype(np.float32, copy=False),
        prior_distribution=prior_distribution.astype(np.float32, copy=False),
        metadata=_label_smoothing_metadata(cfg, labels=labels, classes=class_values, prior_distribution=prior_distribution),
    )


def source_label_prior(
    source_labels: Sequence[Any] | np.ndarray,
    *,
    classes: Sequence[Any] | np.ndarray | None = None,
    prior: str | None = "uniform",
    epsilon: float | str = 1e-12,
) -> np.ndarray:
    """Return a source-only class prior distribution."""

    labels = _vector(source_labels, name="source_labels")
    class_values = _classes(labels, classes)
    mode = normalize_label_smoothing_prior(prior)
    if mode == "uniform":
        values = np.full(class_values.shape[0], 1.0 / class_values.shape[0], dtype=float)
    else:
        counts = np.asarray([np.count_nonzero(labels == class_label) for class_label in class_values.tolist()], dtype=float)
        values = counts / float(np.sum(counts))
    return _normalize_probability_rows(values[None, :], epsilon=_positive_float(epsilon, name="epsilon"))[0]


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
    labels = _vector(source_labels, name="source_labels")
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
        key_array = np.asarray(keys, dtype=object)
        for key in tuple(dict.fromkeys(keys)):
            group_indices = np.flatnonzero(key_array == key)
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


def normalize_label_smoothing_prior(value: str | None) -> str:
    """Normalize label-smoothing prior aliases."""

    normalized = "uniform" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"balanced": "uniform", "flat": "uniform", "frequency": "empirical", "counts": "empirical"}.get(normalized, normalized)
    if normalized not in LABEL_SMOOTHING_PRIORS:
        raise ValueError(f"Unknown label smoothing prior {value!r}.")
    return normalized


def _coerce_config(config: SourceBalanceConfig | Mapping[str, Any]) -> SourceBalanceConfig:
    if isinstance(config, SourceBalanceConfig):
        return config
    return source_balance_config(**dict(config))


def _coerce_label_smoothing_config(config: SourceLabelSmoothingConfig | Mapping[str, Any]) -> SourceLabelSmoothingConfig:
    if isinstance(config, SourceLabelSmoothingConfig):
        return config
    return source_label_smoothing_config(**dict(config))


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


def _classes(labels: np.ndarray, classes: Sequence[Any] | np.ndarray | None) -> np.ndarray:
    if classes is None:
        return np.asarray(tuple(dict.fromkeys(labels.tolist())), dtype=object)
    class_values = np.asarray(classes, dtype=object).reshape(-1)
    if class_values.shape[0] < 1:
        raise ValueError("classes must contain at least one value.")
    if len(set(class_values.tolist())) != class_values.shape[0]:
        raise ValueError("classes must be unique.")
    missing = sorted({label for label in labels.tolist() if label not in set(class_values.tolist())}, key=repr)
    if missing:
        raise ValueError(f"source_labels contain labels absent from classes: {missing}.")
    return class_values


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


def _label_smoothing_metadata(cfg: SourceLabelSmoothingConfig, *, labels: np.ndarray, classes: np.ndarray, prior_distribution: np.ndarray) -> dict[str, Any]:
    unique, counts = np.unique(labels.astype(str), return_counts=True)
    return {
        "source_label_smoothing": True,
        "source_label_smoothing_protocol": SOURCE_LABEL_SMOOTHING_PROTOCOL,
        "source_label_smoothing_protocol_category": SOURCE_LABEL_SMOOTHING_CATEGORY,
        "source_label_smoothing_uses_source_labels": True,
        "source_label_smoothing_uses_heldout_features": False,
        "source_label_smoothing_uses_heldout_labels": False,
        "source_label_smoothing_valid_for_strict_source_only": True,
        "source_label_smoothing_valid_for_benchmark": True,
        "source_label_smoothing_n_rows": int(labels.shape[0]),
        "source_label_smoothing_n_classes": int(classes.shape[0]),
        "source_label_smoothing_smoothing": float(cfg.smoothing),
        "source_label_smoothing_prior": cfg.prior,
        "source_label_smoothing_class_counts": "|".join(f"{label}:{int(count)}" for label, count in zip(unique, counts, strict=True)),
        "source_label_smoothing_prior_distribution": "|".join(f"{label}:{float(prob):.12g}" for label, prob in zip(classes.tolist(), prior_distribution, strict=True)),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _vector(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=object).reshape(-1)
    if vector.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one value.")
    return vector


def _domain_vector(values: Sequence[Hashable] | np.ndarray | None, *, expected_length: int) -> np.ndarray:
    if values is None:
        return np.full(expected_length, "source", dtype=object)
    vector = np.asarray(values, dtype=object).reshape(-1)
    if vector.shape[0] != expected_length:
        raise ValueError("source_domains must contain one value per source row.")
    return vector


def _normalize_probability_rows(values: np.ndarray, *, epsilon: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probability rows must be finite and non-negative.")
    matrix = np.maximum(matrix, float(epsilon))
    return matrix / np.sum(matrix, axis=1, keepdims=True)


def _nonnegative_int(value: int | str, *, name: str) -> int:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(parsed)


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _unit_interval_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed
