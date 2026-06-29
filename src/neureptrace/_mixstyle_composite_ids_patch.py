"""Preserve composite labels and domain IDs in feature-matrix MixStyle."""

from __future__ import annotations

import importlib
from collections.abc import Hashable, Iterable, Mapping, Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = '_neureptrace_mixstyle_composite_ids_patch_installed'


def _object_value_vector(values: Iterable[Any]) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _atomic_value_vector(values: Any, *, expected_length: int, name: str) -> np.ndarray:
    """Return a 1-D object vector without flattening composite row values."""

    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        vector = _object_value_vector([array.item()])
    elif array.ndim == 1:
        if array.shape[0] == expected_length:
            vector = _object_value_vector(array.tolist())
        elif expected_length == 1:
            vector = _object_value_vector([tuple(array.tolist())])
        else:
            vector = _object_value_vector(array.tolist())
    else:
        rows = array.reshape(array.shape[0], -1)
        if rows.shape[1] == 1:
            vector = _object_value_vector(rows[:, 0].tolist())
        else:
            vector = _object_value_vector(tuple(row.tolist()) for row in rows)

    if vector.shape[0] != expected_length:
        raise ValueError(f'{name} must contain one value per source row: {vector.shape[0]} != {expected_length}.')
    return vector


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int) -> np.ndarray:
    return _atomic_value_vector(values, expected_length=expected_length, name='source_labels')


def _domain_vector(values: Sequence[Hashable] | np.ndarray, *, expected_length: int) -> np.ndarray:
    vector = _atomic_value_vector(values, expected_length=expected_length, name='source_domains')
    for domain in vector.tolist():
        try:
            hash(domain)
        except TypeError as exc:
            raise ValueError(f'source_domains must be hashable; got {domain!r}.') from exc
    return vector


def _equal_mask(values: np.ndarray, value: Any) -> np.ndarray:
    return np.asarray([candidate == value for candidate in values.tolist()], dtype=bool)


def _unique_values(values: np.ndarray) -> tuple[Any, ...]:
    unique: list[Any] = []
    for value in values.tolist():
        if not any(existing == value for existing in unique):
            unique.append(value)
    return tuple(unique)


def _value_key(value: Any) -> Any:
    try:
        hash(value)
    except TypeError:
        if isinstance(value, np.ndarray):
            return tuple(_value_key(item) for item in value.tolist())
        if isinstance(value, list):
            return tuple(_value_key(item) for item in value)
        if isinstance(value, tuple):
            return tuple(_value_key(item) for item in value)
        if isinstance(value, dict):
            return tuple(sorted((_value_key(key), _value_key(item)) for key, item in value.items()))
        return repr(value)
    return value


def install() -> None:
    """Patch feature-matrix MixStyle label/domain handling for composite IDs."""

    mixstyle = importlib.import_module('neureptrace.decoding.mixstyle')
    original_augment = mixstyle.augment_source_mixstyle
    if getattr(original_augment, _PATCH_MARKER, False):
        return

    def _domain_statistics(features: np.ndarray, domains: np.ndarray) -> dict[Hashable, tuple[np.ndarray, np.ndarray]]:
        stats = {}
        for domain in _unique_values(domains):
            rows = features[_equal_mask(domains, domain)]
            stats[domain] = mixstyle._mean_std(rows)
        return stats

    def _label_domain_statistics(features: np.ndarray, labels: np.ndarray, domains: np.ndarray) -> dict[tuple[Any, Hashable], tuple[np.ndarray, np.ndarray]]:
        stats = {}
        unique_domains = _unique_values(domains)
        for label in _unique_values(labels):
            label_mask = _equal_mask(labels, label)
            for domain in unique_domains:
                mask = label_mask & _equal_mask(domains, domain)
                if np.any(mask):
                    stats[(_value_key(label), domain)] = mixstyle._mean_std(features[mask])
        return stats

    def _stats_for_row(
        label_domain_stats: Mapping[tuple[Any, Hashable], tuple[np.ndarray, np.ndarray]],
        domain_stats: Mapping[Hashable, tuple[np.ndarray, np.ndarray]],
        label: Any,
        domain: Hashable,
    ) -> tuple[np.ndarray, np.ndarray]:
        return label_domain_stats.get((_value_key(label), domain), domain_stats[domain])

    @wraps(original_augment)
    def augment_source_mixstyle(
        source_features: Sequence[Sequence[float]] | np.ndarray,
        source_labels: Sequence[Any] | np.ndarray,
        source_domains: Sequence[Hashable] | np.ndarray,
        *,
        augmentations_per_row: int | str = mixstyle.DEFAULT_MIXSTYLE_AUGMENTATIONS_PER_ROW,
        alpha: float | str = mixstyle.DEFAULT_MIXSTYLE_ALPHA,
        random_state: int | str | None = mixstyle.DEFAULT_MIXSTYLE_RANDOM_STATE,
        domain_pairing: str = 'shuffle',
        preserve_domain_mean: bool = False,
        class_conditional: bool = False,
        include_original: bool = True,
    ):
        cfg = mixstyle.source_mixstyle_config(
            augmentations_per_row=augmentations_per_row,
            alpha=alpha,
            random_state=random_state,
            domain_pairing=domain_pairing,
            preserve_domain_mean=preserve_domain_mean,
            class_conditional=class_conditional,
            include_original=include_original,
        )
        features = mixstyle._feature_matrix(source_features, name='source_features')
        labels = _label_vector(source_labels, expected_length=features.shape[0])
        domains = _domain_vector(source_domains, expected_length=features.shape[0])
        rng = np.random.default_rng(cfg.random_state)

        domain_stats = _domain_statistics(features, domains)
        domain_names = tuple(domain_stats)
        if len(domain_names) < 2 and cfg.augmentations_per_row > 0:
            raise ValueError('Source MixStyle requires at least two source domains when augmentations_per_row > 0.')
        label_domain_stats = _label_domain_statistics(features, labels, domains) if cfg.class_conditional else {}
        deterministic_partner_order = mixstyle._deterministic_partner_order(domain_stats)

        synthetic_features: list[np.ndarray] = []
        synthetic_labels: list[Any] = []
        synthetic_domains: list[str] = []
        for row, label, domain in zip(features, labels, domains, strict=True):
            for _ in range(cfg.augmentations_per_row):
                partner = mixstyle._choose_partner_domain(
                    domain,
                    domain_names=domain_names,
                    deterministic_partner_order=deterministic_partner_order,
                    pairing=cfg.domain_pairing,
                    rng=rng,
                )
                source_stats = _stats_for_row(label_domain_stats, domain_stats, label, domain)
                partner_stats = _stats_for_row(label_domain_stats, domain_stats, label, partner)
                synthetic = mixstyle._mixstyle_row(
                    row,
                    source_stats=source_stats,
                    partner_stats=partner_stats,
                    mix=float(rng.beta(cfg.alpha, cfg.alpha)),
                    preserve_domain_mean=cfg.preserve_domain_mean,
                )
                synthetic_features.append(synthetic)
                synthetic_labels.append(label)
                synthetic_domains.append(f'mixstyle:{domain}->{partner}')

        feature_blocks = []
        label_blocks = []
        domain_blocks = []
        mask_blocks = []
        if cfg.include_original:
            feature_blocks.append(features)
            label_blocks.append(labels)
            domain_blocks.append(domains)
            mask_blocks.append(np.zeros(features.shape[0], dtype=bool))
        if synthetic_features:
            feature_blocks.append(np.vstack(synthetic_features))
            label_blocks.append(_object_value_vector(synthetic_labels))
            domain_blocks.append(_object_value_vector(synthetic_domains))
            mask_blocks.append(np.ones(len(synthetic_features), dtype=bool))
        if not feature_blocks:
            raise ValueError('No rows would be returned; enable include_original or request augmentations_per_row > 0.')

        output_features = np.vstack(feature_blocks).astype(np.float32, copy=False)
        output_labels = np.concatenate(label_blocks).astype(object, copy=False)
        output_domains = np.concatenate(domain_blocks).astype(object, copy=False)
        synthetic_mask = np.concatenate(mask_blocks)
        metadata = mixstyle._metadata(
            n_source_rows=features.shape[0],
            n_output_rows=output_features.shape[0],
            n_synthetic_rows=int(np.count_nonzero(synthetic_mask)),
            n_domains=len(domain_names),
            n_classes=len(_unique_values(labels)),
            feature_dim=features.shape[1],
            augmentations_per_row=cfg.augmentations_per_row,
            alpha=cfg.alpha,
            random_state=cfg.random_state,
            domain_pairing=cfg.domain_pairing,
            preserve_domain_mean=cfg.preserve_domain_mean,
            class_conditional=cfg.class_conditional,
            include_original=cfg.include_original,
        )
        return mixstyle.SourceMixStyleResult(
            features=output_features,
            labels=output_labels,
            domains=output_domains,
            synthetic_mask=synthetic_mask,
            metadata=metadata,
        )

    setattr(augment_source_mixstyle, _PATCH_MARKER, True)
    mixstyle._label_vector = _label_vector
    mixstyle._domain_vector = _domain_vector
    mixstyle._domain_statistics = _domain_statistics
    mixstyle._label_domain_statistics = _label_domain_statistics
    mixstyle._stats_for_row = _stats_for_row
    mixstyle.augment_source_mixstyle = augment_source_mixstyle


__all__ = ['install']
