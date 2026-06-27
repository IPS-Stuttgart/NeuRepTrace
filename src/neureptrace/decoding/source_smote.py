"""Strict source-only SMOTE-style interpolation.

This module creates same-class interpolated source feature rows.  It is intended
as a dependency-light domain-generalization baseline for M/EEG feature matrices.
Synthetic rows keep the sampled source class label and are generated only from
source rows, so the protocol is strict source-only.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_SMOTE_AUGMENTATION = "source_smote"
SOURCE_SMOTE_PROTOCOL = "strict_source_only_smote_interpolation"
SOURCE_SMOTE_CATEGORY = "1_strict_source_only"


@dataclass(frozen=True, slots=True)
class SourceSmoteConfig:
    """Configuration for source-only same-class interpolation."""

    synthetic_per_class: int = 0
    cross_domain_partner: bool = True
    preserve_original: bool = True
    random_state: int | None = 13
    jitter_std: float = 0.0

    @property
    def enabled(self) -> bool:
        """Whether this config requests synthetic rows."""

        return self.synthetic_per_class > 0


@dataclass(frozen=True, slots=True)
class SourceSmoteResult:
    """Augmented source rows and interpolation provenance."""

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

def augment_source_with_smote(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    *,
    source_domains: Sequence[Hashable] | np.ndarray | None = None,
    config: SourceSmoteConfig | Mapping[str, Any] | None = None,
) -> SourceSmoteResult:
    """Append same-class source interpolation rows.

    Parameters
    ----------
    source_features:
        Source feature matrix.
    source_labels:
        One source label per row.  Synthetic rows inherit the sampled class label.
    source_domains:
        Optional source-domain identifiers.  When ``cross_domain_partner=True``,
        interpolation partners are drawn from another source domain when possible.
    config:
        SMOTE options.  Mappings are normalized through :func:`source_smote_config`.
    """

    cfg = source_smote_config() if config is None else _coerce_config(config)
    features = _feature_matrix(source_features, name="source_features")
    labels = _label_vector(source_labels, expected_length=features.shape[0], name="source_labels")
    domains = _domain_vector(source_domains, expected_length=features.shape[0])

    if not cfg.enabled:
        metadata = _metadata(cfg, n_source_rows=features.shape[0], n_synthetic_rows=0, n_classes=np.unique(labels).shape[0], n_source_domains=np.unique(domains).shape[0], feature_dim=features.shape[1])
        return SourceSmoteResult(
            features=features.astype(np.float32, copy=False),
            labels=labels.copy(),
            synthetic_mask=np.zeros(features.shape[0], dtype=bool),
            content_indices=np.empty(0, dtype=int),
            partner_indices=np.empty(0, dtype=int),
            lambdas=np.empty(0, dtype=float),
            metadata=metadata,
        )

    rng = np.random.default_rng(cfg.random_state)
    synthetic_rows: list[np.ndarray] = []
    synthetic_labels: list[Any] = []
    content_indices: list[int] = []
    partner_indices: list[int] = []
    lambdas: list[float] = []

    for class_label in tuple(dict.fromkeys(labels.tolist())):
        class_indices = np.flatnonzero(labels == class_label)
        if class_indices.size == 0:
            continue
        for _ in range(cfg.synthetic_per_class):
            content_index = int(rng.choice(class_indices))
            partner_pool = class_indices[class_indices != content_index] if class_indices.size > 1 else class_indices
            if cfg.cross_domain_partner and partner_pool.size > 0:
                cross_pool = partner_pool[domains[partner_pool] != domains[content_index]]
                if cross_pool.size:
                    partner_pool = cross_pool
            if partner_pool.size == 0:
                partner_pool = np.asarray([content_index], dtype=int)
            partner_index = int(rng.choice(partner_pool))
            lam = float(rng.random())
            row = interpolate_rows(features[content_index], features[partner_index], lam)
            if cfg.jitter_std > 0.0:
                row = row + rng.normal(0.0, cfg.jitter_std, size=row.shape[0])
            synthetic_rows.append(row)
            synthetic_labels.append(class_label)
            content_indices.append(content_index)
            partner_indices.append(partner_index)
            lambdas.append(lam)

    synthetic_features = np.vstack(synthetic_rows).astype(np.float32, copy=False) if synthetic_rows else np.empty((0, features.shape[1]), dtype=np.float32)
    synthetic_label_array = np.asarray(synthetic_labels, dtype=labels.dtype if labels.dtype != object else object)
    if cfg.preserve_original:
        output_features = np.vstack([features, synthetic_features]).astype(np.float32, copy=False)
        output_labels = np.concatenate([labels, synthetic_label_array])
        synthetic_mask = np.concatenate([np.zeros(features.shape[0], dtype=bool), np.ones(synthetic_features.shape[0], dtype=bool)])
    else:
        output_features = synthetic_features
        output_labels = synthetic_label_array
        synthetic_mask = np.ones(synthetic_features.shape[0], dtype=bool)

    metadata = _metadata(cfg, n_source_rows=features.shape[0], n_synthetic_rows=synthetic_features.shape[0], n_classes=np.unique(labels).shape[0], n_source_domains=np.unique(domains).shape[0], feature_dim=features.shape[1])
    return SourceSmoteResult(
        features=output_features,
        labels=output_labels,
        synthetic_mask=synthetic_mask,
        content_indices=np.asarray(content_indices, dtype=int),
        partner_indices=np.asarray(partner_indices, dtype=int),
        lambdas=np.asarray(lambdas, dtype=float),
        metadata=metadata,
    )


def interpolate_rows(content_row: Sequence[float] | np.ndarray, partner_row: Sequence[float] | np.ndarray, lam: float | str) -> np.ndarray:
    """Return a convex interpolation of two feature rows."""

    left = np.asarray(content_row, dtype=float).reshape(-1)
    right = np.asarray(partner_row, dtype=float).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("content_row and partner_row must be non-empty vectors with the same shape.")
    weight = _unit_interval_float(lam, name="lam")
    return (left + weight * (right - left)).astype(np.float32, copy=False)


def source_smote_config(
    *,
    synthetic_per_class: int | str = 0,
    cross_domain_partner: bool = True,
    preserve_original: bool = True,
    random_state: int | str | None = 13,
    jitter_std: float | str = 0.0,
) -> SourceSmoteConfig:
    """Normalize public source-SMOTE options."""

    return SourceSmoteConfig(
        synthetic_per_class=_nonnegative_int(synthetic_per_class, name="synthetic_per_class"),
        cross_domain_partner=bool(cross_domain_partner),
        preserve_original=bool(preserve_original),
        random_state=None if random_state in {None, "", "none", "None"} else _nonnegative_int(random_state, name="random_state"),
        jitter_std=_nonnegative_float(jitter_std, name="jitter_std"),
    )


def _coerce_config(config: SourceSmoteConfig | Mapping[str, Any]) -> SourceSmoteConfig:
    if isinstance(config, SourceSmoteConfig):
        return config
    return source_smote_config(**dict(config))


def _metadata(cfg: SourceSmoteConfig, *, n_source_rows: int, n_synthetic_rows: int, n_classes: int, n_source_domains: int, feature_dim: int) -> dict[str, Any]:
    return {
        "source_smote": bool(cfg.enabled),
        "source_smote_protocol": SOURCE_SMOTE_PROTOCOL,
        "source_smote_protocol_category": SOURCE_SMOTE_CATEGORY,
        "source_smote_method": SOURCE_SMOTE_AUGMENTATION,
        "source_smote_uses_source_features": True,
        "source_smote_uses_source_labels": True,
        "source_smote_uses_source_domains": True,
        "source_smote_uses_heldout_features": False,
        "source_smote_uses_heldout_labels": False,
        "source_smote_valid_for_strict_source_only": True,
        "source_smote_valid_for_unlabeled_target_adaptation": True,
        "source_smote_valid_for_benchmark": True,
        "source_smote_n_source_rows": int(n_source_rows),
        "source_smote_n_synthetic_rows": int(n_synthetic_rows),
        "source_smote_n_output_rows": int(n_source_rows + n_synthetic_rows if cfg.preserve_original else n_synthetic_rows),
        "source_smote_n_classes": int(n_classes),
        "source_smote_n_source_domains": int(n_source_domains),
        "source_smote_feature_dim": int(feature_dim),
        "source_smote_synthetic_per_class": int(cfg.synthetic_per_class),
        "source_smote_cross_domain_partner": bool(cfg.cross_domain_partner),
        "source_smote_preserve_original": bool(cfg.preserve_original),
        "source_smote_random_state": "" if cfg.random_state is None else int(cfg.random_state),
        "source_smote_jitter_std": float(cfg.jitter_std),
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


def _unit_interval_float(value: float | str, *, name: str) -> float:
    parsed = _float_value(value, name=name)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
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
