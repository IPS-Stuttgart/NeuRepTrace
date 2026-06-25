"""Preserve tuple-valued Source MixStyle labels and domain IDs atomically."""

from __future__ import annotations

import importlib
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np

_INSTALLED = False


def _object_vector_from_items(items: Sequence[Any]) -> np.ndarray:
    vector = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        vector[index] = item
    return vector


def _row_tuple(row: Any) -> tuple[Any, ...]:
    return tuple(np.asarray(row, dtype=object).reshape(-1).tolist())


def _as_atomic_vector(
    values: Any,
    *,
    expected_length: int,
    name: str,
    preserve_scalar_dtype: bool,
) -> np.ndarray:
    object_array = np.asarray(values, dtype=object)
    try:
        scalar_array = np.asarray(values)
    except ValueError:
        scalar_array = object_array

    if object_array.ndim == 0:
        vector = _object_vector_from_items([object_array.item()])
    elif scalar_array.ndim == 1 and preserve_scalar_dtype:
        vector = scalar_array.reshape(-1).copy()
    elif object_array.ndim == 1:
        vector = _object_vector_from_items(object_array.tolist())
    elif scalar_array.ndim == 2 and 1 in scalar_array.shape and preserve_scalar_dtype:
        vector = scalar_array.reshape(-1).copy()
    elif object_array.ndim == 2 and 1 in object_array.shape:
        vector = _object_vector_from_items(object_array.reshape(-1).tolist())
    elif object_array.shape[0] == expected_length:
        vector = _object_vector_from_items([_row_tuple(object_array[index]) for index in range(object_array.shape[0])])
    else:
        vector = object_array.reshape(-1)

    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one value per source row: {vector.shape[0]} != {expected_length}.")
    return vector


def _hashable_domain_vector(values: Any, *, expected_length: int) -> np.ndarray:
    vector = _as_atomic_vector(
        values,
        expected_length=expected_length,
        name="source_domains",
        preserve_scalar_dtype=False,
    )
    for domain in vector.tolist():
        try:
            hash(domain)
        except TypeError as exc:
            raise ValueError(f"source_domains must be hashable; got {domain!r}.") from exc
    return vector


def _label_vector(values: Any, *, expected_length: int, name: str) -> np.ndarray:
    return _as_atomic_vector(
        values,
        expected_length=expected_length,
        name=name,
        preserve_scalar_dtype=True,
    )


def _domain_equal_mask(domains: np.ndarray, domain: Hashable) -> np.ndarray:
    return np.asarray([candidate == domain for candidate in domains.tolist()], dtype=bool)


def install() -> None:
    """Install atomic tuple-label/domain handling for Source MixStyle."""

    global _INSTALLED
    if _INSTALLED:
        return

    module = importlib.import_module("neureptrace.decoding.source_mixstyle")

    def _domain_stats(features: np.ndarray, domains: np.ndarray, unique_domains: Sequence[Hashable]) -> dict[Hashable, Any]:
        stats: dict[Hashable, Any] = {}
        for domain in unique_domains:
            domain_features = features[_domain_equal_mask(domains, domain)]
            if domain_features.shape[0] < 1:
                raise ValueError(f"Source domain {domain!r} has no rows.")
            mean = np.mean(domain_features, axis=0)
            scale = np.std(domain_features, axis=0, ddof=0)
            scale = np.maximum(scale, module._MIN_SCALE)
            stats[domain] = module._DomainStyleStats(domain_id=domain, mean=mean, scale=scale, n_rows=domain_features.shape[0])
        return stats

    def _synthetic_vector(items: Sequence[Any], *, like: np.ndarray) -> np.ndarray:
        if like.dtype == object:
            return _object_vector_from_items(items)
        return np.asarray(items, dtype=like.dtype)

    def augment_source_domains_mixstyle(
        source_features: Sequence[Sequence[float]] | np.ndarray,
        source_labels: Sequence[Any] | np.ndarray,
        source_domains: Sequence[Hashable] | np.ndarray,
        *,
        config: Any = None,
    ):
        cfg = module._coerce_config(config)
        features = module._feature_matrix(source_features, name="source_features")
        labels = _label_vector(source_labels, expected_length=features.shape[0], name="source_labels")
        domains = _hashable_domain_vector(source_domains, expected_length=features.shape[0])
        unique_domains = tuple(dict.fromkeys(domains.tolist()))
        if len(unique_domains) < 2 and cfg.mixes_per_row > 0:
            raise ValueError("MixStyle source-domain augmentation requires at least two source domains.")
        stats = _domain_stats(features, domains, unique_domains)

        if cfg.mixes_per_row == 0:
            return module._original_only_result(features, labels, domains, cfg=cfg, n_domains=len(unique_domains))

        rng = np.random.default_rng(cfg.random_state)
        synthetic_features: list[np.ndarray] = []
        synthetic_labels: list[Any] = []
        synthetic_domains: list[Hashable] = []
        synthetic_lambdas: list[float] = []
        synthetic_partner_domains: list[Hashable] = []

        for row_index, row in enumerate(features):
            own_domain = domains[row_index]
            own_stats = stats[own_domain]
            partner_pool = tuple(domain for domain in unique_domains if domain != own_domain)
            for _ in range(cfg.mixes_per_row):
                partner_domain = partner_pool[int(rng.integers(0, len(partner_pool)))]
                partner_stats = stats[partner_domain]
                lam = float(rng.beta(cfg.alpha, cfg.alpha))
                mixed = module.mixstyle_row(
                    row,
                    source_stats=own_stats,
                    partner_stats=partner_stats,
                    lam=lam,
                    style_strength=cfg.style_strength,
                )
                synthetic_features.append(mixed)
                synthetic_labels.append(labels[row_index])
                synthetic_domains.append(own_domain)
                synthetic_lambdas.append(lam)
                synthetic_partner_domains.append(partner_domain)

        synthetic_matrix = np.vstack(synthetic_features).astype(np.float32, copy=False)
        synthetic_label_vector = _synthetic_vector(synthetic_labels, like=labels)
        synthetic_domain_vector = _object_vector_from_items(synthetic_domains)
        if cfg.include_original:
            output_features = np.vstack([features, synthetic_matrix]).astype(np.float32, copy=False)
            output_labels = np.concatenate([labels, synthetic_label_vector])
            output_domains = np.concatenate([domains, synthetic_domain_vector])
            synthetic_mask = np.concatenate([np.zeros(features.shape[0], dtype=bool), np.ones(synthetic_matrix.shape[0], dtype=bool)])
            sample_weight = np.concatenate(
                [
                    np.ones(features.shape[0], dtype=float),
                    np.full(synthetic_matrix.shape[0], float(cfg.synthetic_weight), dtype=float),
                ]
            )
        else:
            output_features = synthetic_matrix
            output_labels = synthetic_label_vector
            output_domains = synthetic_domain_vector
            synthetic_mask = np.ones(synthetic_matrix.shape[0], dtype=bool)
            sample_weight = np.full(synthetic_matrix.shape[0], float(cfg.synthetic_weight), dtype=float)

        metadata = module._metadata(
            cfg=cfg,
            n_input_rows=features.shape[0],
            n_output_rows=output_features.shape[0],
            n_synthetic=int(np.count_nonzero(synthetic_mask)),
            n_domains=len(unique_domains),
            feature_dim=features.shape[1],
            synthetic_lambdas=synthetic_lambdas,
            synthetic_partner_domains=synthetic_partner_domains,
        )
        return module.SourceMixStyleResult(
            features=output_features,
            labels=output_labels,
            sample_weight=sample_weight,
            domain_ids=output_domains,
            synthetic_mask=synthetic_mask,
            metadata=metadata,
        )

    module._label_vector = _label_vector
    module._domain_vector = _hashable_domain_vector
    module._domain_stats = _domain_stats
    module.augment_source_domains_mixstyle = augment_source_domains_mixstyle
    _INSTALLED = True
