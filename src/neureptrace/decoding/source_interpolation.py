"""Strict source-only interpolation augmentation.

This module creates synthetic source rows by interpolating between same-class
source rows.  Optional source-domain identifiers can be used to prefer partners
from another source domain, but held-out target rows and labels are not accepted.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_INTERPOLATION_PROTOCOL = "strict_source_only_interpolation_augmentation"
SOURCE_INTERPOLATION_CATEGORY = "1_strict_source_only"
PAIR_MODES = ("same_class", "same_class_cross_domain")
DEFAULT_SYNTHETIC_PER_CLASS = 0
DEFAULT_ALPHA = 1.0


@dataclass(frozen=True, slots=True)
class SourceInterpolationConfig:
    """Configuration for source-only interpolation augmentation."""

    synthetic_per_class: int = DEFAULT_SYNTHETIC_PER_CLASS
    pair_mode: str = "same_class"
    alpha: float = DEFAULT_ALPHA
    preserve_original: bool = True
    random_state: int | None = 13

    @property
    def enabled(self) -> bool:
        """Whether this configuration generates synthetic rows."""

        return self.synthetic_per_class > 0


@dataclass(frozen=True, slots=True)
class SourceInterpolationResult:
    """Augmented source rows and provenance."""

    features: np.ndarray
    labels: np.ndarray
    synthetic_mask: np.ndarray
    content_indices: np.ndarray
    partner_indices: np.ndarray
    lambdas: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_synthetic(self) -> int:
        """Number of generated rows in the returned matrix."""

        return int(np.sum(self.synthetic_mask))


# pylint: disable-next=too-many-locals
def augment_source_with_interpolation(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    *,
    source_domains: Sequence[Hashable] | np.ndarray | None = None,
    config: SourceInterpolationConfig | Mapping[str, Any] | None = None,
) -> SourceInterpolationResult:
    """Append source-only interpolated rows.

    Synthetic rows inherit the class label of the sampled source content row.
    Partner rows are sampled from the same source class; in cross-domain mode a
    different source domain is preferred whenever possible.
    """

    cfg = source_interpolation_config() if config is None else _coerce_config(config)
    features = _feature_matrix(source_features, name="source_features")
    labels = _value_vector(source_labels, expected_length=features.shape[0], name="source_labels")
    domains = _domain_vector(source_domains, expected_length=features.shape[0])
    classes = tuple(dict.fromkeys(labels.tolist()))
    domain_count = len(tuple(dict.fromkeys(domains.tolist())))

    if not cfg.enabled:
        metadata = _metadata(cfg, n_source_rows=features.shape[0], n_synthetic_rows=0, n_classes=len(classes), n_source_domains=domain_count, feature_dim=features.shape[1])
        return SourceInterpolationResult(
            features=features.astype(np.float32, copy=False),
            labels=labels.copy(),
            synthetic_mask=np.zeros(features.shape[0], dtype=bool),
            content_indices=np.empty(0, dtype=int),
            partner_indices=np.empty(0, dtype=int),
            lambdas=np.empty(0, dtype=float),
            metadata=metadata,
        )

    rng = np.random.default_rng(cfg.random_state)
    rows: list[np.ndarray] = []
    row_labels: list[Any] = []
    content_indices: list[int] = []
    partner_indices: list[int] = []
    lambdas: list[float] = []
    for class_label in classes:
        class_indices = np.flatnonzero(labels == class_label)
        if class_indices.size == 0:
            continue
        for _ in range(cfg.synthetic_per_class):
            content_index = int(rng.choice(class_indices))
            partners = _partner_pool(class_indices, domains, content_index=content_index, pair_mode=cfg.pair_mode)
            partner_index = int(rng.choice(partners))
            lam = float(rng.beta(cfg.alpha, cfg.alpha))
            rows.append(interpolate_rows(features[content_index], features[partner_index], lam))
            row_labels.append(class_label)
            content_indices.append(content_index)
            partner_indices.append(partner_index)
            lambdas.append(lam)

    synthetic_features = np.vstack(rows).astype(np.float32, copy=False) if rows else np.empty((0, features.shape[1]), dtype=np.float32)
    synthetic_labels = _object_array(row_labels)
    if cfg.preserve_original:
        output_features = np.vstack([features, synthetic_features]).astype(np.float32, copy=False)
        output_labels = np.concatenate([labels, synthetic_labels])
        synthetic_mask = np.concatenate([np.zeros(features.shape[0], dtype=bool), np.ones(synthetic_features.shape[0], dtype=bool)])
    else:
        output_features = synthetic_features
        output_labels = synthetic_labels
        synthetic_mask = np.ones(synthetic_features.shape[0], dtype=bool)

    metadata = _metadata(cfg, n_source_rows=features.shape[0], n_synthetic_rows=synthetic_features.shape[0], n_classes=len(classes), n_source_domains=domain_count, feature_dim=features.shape[1])
    return SourceInterpolationResult(
        features=output_features,
        labels=output_labels,
        synthetic_mask=synthetic_mask,
        content_indices=np.asarray(content_indices, dtype=int),
        partner_indices=np.asarray(partner_indices, dtype=int),
        lambdas=np.asarray(lambdas, dtype=float),
        metadata=metadata,
    )


def interpolate_rows(content: Sequence[float] | np.ndarray, partner: Sequence[float] | np.ndarray, lam: float | str) -> np.ndarray:
    """Return a convex interpolation of two feature rows."""

    left = np.asarray(content, dtype=float).reshape(-1)
    right = np.asarray(partner, dtype=float).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"content and partner must have the same shape: {left.shape} != {right.shape}.")
    if left.size == 0 or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("content and partner must be finite non-empty vectors.")
    weight = _unit_interval_float(lam, name="lam")
    return (weight * left + (1.0 - weight) * right).astype(np.float32, copy=False)


def source_interpolation_config(
    *,
    synthetic_per_class: int | str = DEFAULT_SYNTHETIC_PER_CLASS,
    pair_mode: str | None = "same_class",
    alpha: float | str = DEFAULT_ALPHA,
    preserve_original: bool | int | str = True,
    random_state: int | str | None = 13,
) -> SourceInterpolationConfig:
    """Normalize public source-interpolation options."""

    return SourceInterpolationConfig(
        synthetic_per_class=_nonnegative_int(synthetic_per_class, name="synthetic_per_class"),
        pair_mode=normalize_pair_mode(pair_mode),
        alpha=_positive_float(alpha, name="alpha"),
        preserve_original=_bool_value(preserve_original, name="preserve_original"),
        random_state=None if random_state in {None, "", "none", "None"} else _nonnegative_int(random_state, name="random_state"),
    )


def normalize_pair_mode(value: str | None) -> str:
    """Normalize interpolation pair-mode aliases."""

    normalized = "same_class" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"class": "same_class", "within_class": "same_class", "cross_domain": "same_class_cross_domain", "class_cross_domain": "same_class_cross_domain"}.get(normalized, normalized)
    if normalized not in PAIR_MODES:
        raise ValueError(f"Unknown source interpolation pair mode {value!r}.")
    return normalized


def _partner_pool(class_indices: np.ndarray, domains: np.ndarray, *, content_index: int, pair_mode: str) -> np.ndarray:
    pool = class_indices[class_indices != content_index]
    if pair_mode == "same_class_cross_domain" and pool.size:
        cross = pool[domains[pool] != domains[content_index]]
        if cross.size:
            pool = cross
    if pool.size == 0:
        pool = np.asarray([content_index], dtype=int)
    return pool


def _metadata(cfg: SourceInterpolationConfig, *, n_source_rows: int, n_synthetic_rows: int, n_classes: int, n_source_domains: int, feature_dim: int) -> dict[str, Any]:
    return {
        "source_interpolation": bool(cfg.enabled),
        "source_interpolation_protocol": SOURCE_INTERPOLATION_PROTOCOL,
        "source_interpolation_protocol_category": SOURCE_INTERPOLATION_CATEGORY,
        "source_interpolation_uses_source_features": True,
        "source_interpolation_uses_source_labels": True,
        "source_interpolation_uses_source_domains": cfg.pair_mode == "same_class_cross_domain",
        "source_interpolation_uses_heldout_features": False,
        "source_interpolation_uses_heldout_labels": False,
        "source_interpolation_valid_for_strict_source_only": True,
        "source_interpolation_valid_for_benchmark": True,
        "source_interpolation_n_source_rows": int(n_source_rows),
        "source_interpolation_n_synthetic_rows": int(n_synthetic_rows),
        "source_interpolation_n_output_rows": int(n_source_rows + n_synthetic_rows if cfg.preserve_original else n_synthetic_rows),
        "source_interpolation_n_classes": int(n_classes),
        "source_interpolation_n_source_domains": int(n_source_domains),
        "source_interpolation_feature_dim": int(feature_dim),
        "source_interpolation_synthetic_per_class": int(cfg.synthetic_per_class),
        "source_interpolation_pair_mode": cfg.pair_mode,
        "source_interpolation_alpha": float(cfg.alpha),
        "source_interpolation_preserve_original": bool(cfg.preserve_original),
        "source_interpolation_random_state": "" if cfg.random_state is None else int(cfg.random_state),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _value_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=object).reshape(-1)
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one value per row: {vector.shape[0]} != {expected_length}.")
    return vector


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


def _object_array(values: Sequence[Any]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        array[index] = value
    return array


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


def _coerce_config(config: SourceInterpolationConfig | Mapping[str, Any]) -> SourceInterpolationConfig:
    if isinstance(config, SourceInterpolationConfig):
        return config
    return source_interpolation_config(**dict(config))
