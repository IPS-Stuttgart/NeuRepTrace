"""Preserve composite labels and domain IDs in source-only MixStyle augmentation."""

from __future__ import annotations

import importlib
from collections.abc import Hashable, Iterable, Sequence
from functools import wraps
from typing import Any

import numpy as np

from . import _mixstyle_composite_ids_patch

_PATCH_MARKER = "_neureptrace_source_mixstyle_tuple_labels_patch_installed"


def _object_value_vector(values: Iterable[Any]) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _atomic_value_vector(values: Any, *, expected_length: int, name: str) -> np.ndarray:
    """Return a 1-D object vector without flattening composite row values.

    NumPy turns ``[("class", "repeat"), ...]`` into a rectangular two-column
    object array.  MixStyle must keep each row label or domain ID atomic rather
    than treating tuple/list entries as independent rows.
    """

    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        vector = _object_value_vector([array.item()])
    elif array.ndim == 1:
        if array.shape[0] == expected_length:
            vector = array.reshape(-1)
        elif expected_length == 1:
            vector = _object_value_vector([tuple(array.tolist())])
        else:
            vector = array.reshape(-1)
    else:
        rows = array.reshape(array.shape[0], -1)
        if rows.shape[1] == 1:
            vector = rows[:, 0].reshape(-1)
        else:
            vector = _object_value_vector(tuple(row.tolist()) for row in rows)

    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one value per source row: {vector.shape[0]} != {expected_length}.")
    return vector


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    return _atomic_value_vector(values, expected_length=expected_length, name=name)


def _domain_vector(values: Sequence[Hashable] | np.ndarray, *, expected_length: int) -> np.ndarray:
    vector = _atomic_value_vector(values, expected_length=expected_length, name="source_domains")
    for domain in vector.tolist():
        try:
            hash(domain)
        except TypeError as exc:
            raise ValueError(f"source_domains must be hashable; got {domain!r}.") from exc
    return vector


def _domain_equal_mask(domains: np.ndarray, domain: Hashable) -> np.ndarray:
    return np.asarray([candidate == domain for candidate in domains.tolist()], dtype=bool)


def install() -> None:
    """Patch MixStyle label/domain handling for composite source identifiers."""

    _mixstyle_composite_ids_patch.install()
    source_mixstyle = importlib.import_module("neureptrace.decoding.source_mixstyle")
    original_augment = source_mixstyle.augment_source_domains_mixstyle
    if getattr(original_augment, _PATCH_MARKER, False):
        return

    def _domain_stats(features: np.ndarray, domains: np.ndarray, unique_domains: Sequence[Hashable]) -> dict[Hashable, Any]:
        stats: dict[Hashable, Any] = {}
        for domain in unique_domains:
            domain_features = features[_domain_equal_mask(domains, domain)]
            if domain_features.shape[0] < 1:
                raise ValueError(f"Source domain {domain!r} has no rows.")
            mean = np.mean(domain_features, axis=0)
            scale = np.std(domain_features, axis=0, ddof=0)
            scale = np.maximum(scale, source_mixstyle._MIN_SCALE)
            stats[domain] = source_mixstyle._DomainStyleStats(domain_id=domain, mean=mean, scale=scale, n_rows=domain_features.shape[0])
        return stats

    @wraps(original_augment)
    def augment_source_domains_mixstyle(
        source_features: Sequence[Sequence[float]] | np.ndarray,
        source_labels: Sequence[Any] | np.ndarray,
        source_domains: Sequence[Hashable] | np.ndarray,
        *,
        config: Any = None,
    ):
        cfg = source_mixstyle._coerce_config(config)
        features = source_mixstyle._feature_matrix(source_features, name="source_features")
        labels = _label_vector(source_labels, expected_length=features.shape[0], name="source_labels")
        domains = _domain_vector(source_domains, expected_length=features.shape[0])
        unique_domains = tuple(dict.fromkeys(domains.tolist()))
        if len(unique_domains) < 2 and cfg.mixes_per_row > 0:
            raise ValueError("MixStyle source-domain augmentation requires at least two source domains.")
        stats = _domain_stats(features, domains, unique_domains)

        if cfg.mixes_per_row == 0:
            return source_mixstyle._original_only_result(features, labels, domains, cfg=cfg, n_domains=len(unique_domains))

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
                mixed = source_mixstyle.mixstyle_row(
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
        synthetic_label_vector = _object_value_vector(synthetic_labels)
        synthetic_domain_vector = _object_value_vector(synthetic_domains)
        if cfg.include_original:
            output_features = np.vstack([features, synthetic_matrix]).astype(np.float32, copy=False)
            output_labels = np.concatenate([labels, synthetic_label_vector])
            output_domains = np.concatenate([domains, synthetic_domain_vector])
            synthetic_mask = np.concatenate([np.zeros(features.shape[0], dtype=bool), np.ones(synthetic_matrix.shape[0], dtype=bool)])
            sample_weight = np.concatenate([
                np.ones(features.shape[0], dtype=float),
                np.full(synthetic_matrix.shape[0], float(cfg.synthetic_weight), dtype=float),
            ])
        else:
            output_features = synthetic_matrix
            output_labels = synthetic_label_vector
            output_domains = synthetic_domain_vector
            synthetic_mask = np.ones(synthetic_matrix.shape[0], dtype=bool)
            sample_weight = np.full(synthetic_matrix.shape[0], float(cfg.synthetic_weight), dtype=float)

        metadata = source_mixstyle._metadata(
            cfg=cfg,
            n_input_rows=features.shape[0],
            n_output_rows=output_features.shape[0],
            n_synthetic=int(np.count_nonzero(synthetic_mask)),
            n_domains=len(unique_domains),
            feature_dim=features.shape[1],
            synthetic_lambdas=synthetic_lambdas,
            synthetic_partner_domains=synthetic_partner_domains,
        )
        return source_mixstyle.SourceMixStyleResult(
            features=output_features,
            labels=output_labels,
            sample_weight=sample_weight,
            domain_ids=output_domains,
            synthetic_mask=synthetic_mask,
            metadata=metadata,
        )

    setattr(augment_source_domains_mixstyle, _PATCH_MARKER, True)
    source_mixstyle._label_vector = _label_vector
    source_mixstyle._domain_vector = _domain_vector
    source_mixstyle._domain_stats = _domain_stats
    source_mixstyle.augment_source_domains_mixstyle = augment_source_domains_mixstyle


__all__ = ["install"]
