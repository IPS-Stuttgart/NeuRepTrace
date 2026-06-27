"""Strict source-only feature scaling augmentation.

This module creates scaled copies of labeled source feature rows.  It is intended
as a dependency-light domain-generalization baseline for M/EEG feature matrices.
Synthetic rows keep the source label of the sampled content row while global-row
or per-feature gain factors are sampled from source-only settings.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_SCALING_AUGMENTATION = "source_feature_scaling"
SOURCE_SCALING_PROTOCOL = "strict_source_only_feature_scaling_augmentation"
SOURCE_SCALING_CATEGORY = "1_strict_source_only"
SCALING_MODES = ("row", "feature")
SCALING_DISTRIBUTIONS = ("lognormal", "uniform", "normal")
DEFAULT_SCALE_STD = 0.1
DEFAULT_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class SourceFeatureScalingConfig:
    """Configuration for strict source-only feature scaling."""

    synthetic_per_class: int = 0
    scale_std: float = DEFAULT_SCALE_STD
    scaling_mode: str = "row"
    distribution: str = "lognormal"
    preserve_original: bool = True
    random_state: int | None = 13
    epsilon: float = DEFAULT_EPSILON

    @property
    def enabled(self) -> bool:
        """Whether synthetic rows should be generated."""

        return self.synthetic_per_class > 0 and self.scale_std > 0.0


@dataclass(frozen=True, slots=True)
class SourceFeatureScalingResult:
    """Augmented source rows and provenance."""

    features: np.ndarray
    labels: np.ndarray
    synthetic_mask: np.ndarray
    content_indices: np.ndarray
    scale_factors: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_synthetic(self) -> int:
        """Number of generated rows in the returned matrix."""

        return int(np.sum(self.synthetic_mask))


# pylint: disable-next=too-many-locals

def augment_source_with_feature_scaling(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    *,
    source_domains: Sequence[Hashable] | np.ndarray | None = None,
    config: SourceFeatureScalingConfig | Mapping[str, Any] | None = None,
) -> SourceFeatureScalingResult:
    """Append gain-scaled source-row copies.

    Parameters
    ----------
    source_features:
        Source feature matrix with rows as trials/windows and columns as features.
    source_labels:
        One source label per row.  Synthetic rows inherit the sampled row label.
    source_domains:
        Optional source-domain ids recorded only for provenance.
    config:
        Scaling options.  Mappings are normalized through
        :func:`source_feature_scaling_config`.
    """

    cfg = source_feature_scaling_config() if config is None else _coerce_config(config)
    features = _feature_matrix(source_features, name="source_features")
    labels = _label_vector(source_labels, expected_length=features.shape[0], name="source_labels")
    domains = _domain_vector(source_domains, expected_length=features.shape[0])

    if not cfg.enabled:
        return SourceFeatureScalingResult(
            features=features.astype(np.float32, copy=False),
            labels=labels.copy(),
            synthetic_mask=np.zeros(features.shape[0], dtype=bool),
            content_indices=np.empty(0, dtype=int),
            scale_factors=np.empty((0, features.shape[1]), dtype=np.float32),
            metadata=_metadata(cfg, features.shape[0], 0, np.unique(labels).shape[0], np.unique(domains).shape[0], features.shape[1]),
        )

    rng = np.random.default_rng(cfg.random_state)
    synthetic_rows: list[np.ndarray] = []
    synthetic_labels: list[Any] = []
    content_indices: list[int] = []
    scale_rows: list[np.ndarray] = []
    for class_label in tuple(dict.fromkeys(labels.tolist())):
        class_indices = np.flatnonzero(labels == class_label)
        if class_indices.size == 0:
            continue
        for _ in range(cfg.synthetic_per_class):
            content_index = int(rng.choice(class_indices))
            scale = sample_scaling_factors(
                features.shape[1],
                scale_std=cfg.scale_std,
                scaling_mode=cfg.scaling_mode,
                distribution=cfg.distribution,
                epsilon=cfg.epsilon,
                rng=rng,
            )
            synthetic_rows.append(features[content_index] * scale)
            synthetic_labels.append(class_label)
            content_indices.append(content_index)
            scale_rows.append(scale)

    synthetic_features = np.vstack(synthetic_rows).astype(np.float32, copy=False) if synthetic_rows else np.empty((0, features.shape[1]), dtype=np.float32)
    synthetic_label_array = np.asarray(synthetic_labels, dtype=labels.dtype if labels.dtype != object else object)
    scale_array = np.vstack(scale_rows).astype(np.float32, copy=False) if scale_rows else np.empty((0, features.shape[1]), dtype=np.float32)
    if cfg.preserve_original:
        output_features = np.vstack([features, synthetic_features]).astype(np.float32, copy=False)
        output_labels = np.concatenate([labels, synthetic_label_array])
        synthetic_mask = np.concatenate([np.zeros(features.shape[0], dtype=bool), np.ones(synthetic_features.shape[0], dtype=bool)])
    else:
        output_features = synthetic_features
        output_labels = synthetic_label_array
        synthetic_mask = np.ones(synthetic_features.shape[0], dtype=bool)

    return SourceFeatureScalingResult(
        features=output_features,
        labels=output_labels,
        synthetic_mask=synthetic_mask,
        content_indices=np.asarray(content_indices, dtype=int),
        scale_factors=scale_array,
        metadata=_metadata(cfg, features.shape[0], synthetic_features.shape[0], np.unique(labels).shape[0], np.unique(domains).shape[0], features.shape[1]),
    )


def sample_scaling_factors(
    n_features: int | str,
    *,
    scale_std: float | str = DEFAULT_SCALE_STD,
    scaling_mode: str = "row",
    distribution: str = "lognormal",
    epsilon: float | str = DEFAULT_EPSILON,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample positive scaling factors for one synthetic row."""

    width = _positive_int(n_features, name="n_features")
    std = _nonnegative_float(scale_std, name="scale_std")
    eps = _positive_float(epsilon, name="epsilon")
    mode = normalize_scaling_mode(scaling_mode)
    dist = normalize_scaling_distribution(distribution)
    generator = np.random.default_rng() if rng is None else rng
    size = 1 if mode == "row" else width
    if dist == "lognormal":
        values = np.exp(generator.normal(0.0, std, size=size))
    elif dist == "uniform":
        values = generator.uniform(max(eps, 1.0 - std), 1.0 + std, size=size)
    elif dist == "normal":
        values = np.maximum(generator.normal(1.0, std, size=size), eps)
    else:  # pragma: no cover - guarded by normalization
        raise ValueError(f"Unhandled scaling distribution {dist!r}.")
    if mode == "row":
        values = np.full(width, float(values[0]), dtype=float)
    return np.asarray(values, dtype=np.float32)


def source_feature_scaling_config(
    *,
    synthetic_per_class: int | str = 0,
    scale_std: float | str = DEFAULT_SCALE_STD,
    scaling_mode: str | None = "row",
    distribution: str | None = "lognormal",
    preserve_original: bool = True,
    random_state: int | str | None = 13,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceFeatureScalingConfig:
    """Normalize public feature-scaling options."""

    return SourceFeatureScalingConfig(
        synthetic_per_class=_nonnegative_int(synthetic_per_class, name="synthetic_per_class"),
        scale_std=_nonnegative_float(scale_std, name="scale_std"),
        scaling_mode=normalize_scaling_mode(scaling_mode),
        distribution=normalize_scaling_distribution(distribution),
        preserve_original=bool(preserve_original),
        random_state=None if random_state in {None, "", "none", "None"} else _nonnegative_int(random_state, name="random_state"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def normalize_scaling_mode(value: str | None) -> str:
    """Normalize scaling-mode aliases."""

    normalized = "row" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"global": "row", "trial": "row", "sample": "row", "column": "feature", "features": "feature"}.get(normalized, normalized)
    if normalized not in SCALING_MODES:
        raise ValueError(f"Unknown scaling_mode {value!r}. Available modes: {', '.join(SCALING_MODES)}.")
    return normalized


def normalize_scaling_distribution(value: str | None) -> str:
    """Normalize scaling-distribution aliases."""

    normalized = "lognormal" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"log_normal": "lognormal", "flat": "uniform", "gaussian": "normal"}.get(normalized, normalized)
    if normalized not in SCALING_DISTRIBUTIONS:
        raise ValueError(f"Unknown scaling distribution {value!r}. Available distributions: {', '.join(SCALING_DISTRIBUTIONS)}.")
    return normalized


def _coerce_config(config: SourceFeatureScalingConfig | Mapping[str, Any]) -> SourceFeatureScalingConfig:
    if isinstance(config, SourceFeatureScalingConfig):
        return config
    return source_feature_scaling_config(**dict(config))


def _metadata(cfg: SourceFeatureScalingConfig, n_source_rows: int, n_synthetic_rows: int, n_classes: int, n_source_domains: int, feature_dim: int) -> dict[str, Any]:
    return {
        "source_feature_scaling": bool(cfg.enabled),
        "source_feature_scaling_protocol": SOURCE_SCALING_PROTOCOL,
        "source_feature_scaling_protocol_category": SOURCE_SCALING_CATEGORY,
        "source_feature_scaling_method": SOURCE_SCALING_AUGMENTATION,
        "source_feature_scaling_uses_source_features": True,
        "source_feature_scaling_uses_source_labels": True,
        "source_feature_scaling_uses_source_domains": True,
        "source_feature_scaling_uses_heldout_features": False,
        "source_feature_scaling_uses_heldout_labels": False,
        "source_feature_scaling_valid_for_strict_source_only": True,
        "source_feature_scaling_valid_for_unlabeled_target_adaptation": True,
        "source_feature_scaling_valid_for_benchmark": True,
        "source_feature_scaling_n_source_rows": int(n_source_rows),
        "source_feature_scaling_n_synthetic_rows": int(n_synthetic_rows),
        "source_feature_scaling_n_output_rows": int(n_source_rows + n_synthetic_rows if cfg.preserve_original else n_synthetic_rows),
        "source_feature_scaling_n_classes": int(n_classes),
        "source_feature_scaling_n_source_domains": int(n_source_domains),
        "source_feature_scaling_feature_dim": int(feature_dim),
        "source_feature_scaling_synthetic_per_class": int(cfg.synthetic_per_class),
        "source_feature_scaling_scale_std": float(cfg.scale_std),
        "source_feature_scaling_scaling_mode": cfg.scaling_mode,
        "source_feature_scaling_distribution": cfg.distribution,
        "source_feature_scaling_preserve_original": bool(cfg.preserve_original),
        "source_feature_scaling_random_state": "" if cfg.random_state is None else int(cfg.random_state),
        "source_feature_scaling_epsilon": float(cfg.epsilon),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = np.asarray(values).reshape(-1)
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one value per feature row: {vector.shape[0]} != {expected_length}.")
    return vector


def _domain_vector(values: Sequence[Hashable] | np.ndarray | None, *, expected_length: int) -> np.ndarray:
    if values is None:
        return np.full(expected_length, "source", dtype=object)
    vector = np.asarray(values, dtype=object).reshape(-1)
    if vector.shape[0] != expected_length:
        raise ValueError(f"source_domains must contain one value per feature row: {vector.shape[0]} != {expected_length}.")
    for value in vector.tolist():
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"source_domains must be hashable; got {value!r}.") from exc
    return vector


def _positive_int(value: int | str, *, name: str) -> int:
    integer = _integer(value, name=name)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _nonnegative_int(value: int | str, *, name: str) -> int:
    integer = _integer(value, name=name)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative.")
    return integer


def _integer(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    return int(parsed)


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = _float_value(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _nonnegative_float(value: float | str, *, name: str) -> float:
    parsed = _float_value(value, name=name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return parsed


def _float_value(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite.")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed
