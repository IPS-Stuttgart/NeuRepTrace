"""Preserve composite labels in source-only MixStyle augmentation."""

from __future__ import annotations

import importlib
from collections.abc import Hashable, Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_mixstyle_tuple_labels_patch_installed"


def _object_value_vector(values: Sequence[Any]) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _atomic_label_vector(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    """Return a 1-D label vector without flattening composite row labels.

    NumPy turns ``[("class", "repeat"), ...]`` into a rectangular two-column
    object array.  Source MixStyle must copy each row label atomically rather
    than treating the tuple/list entries as independent labels.
    """

    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        return _object_value_vector([array.item()])
    if array.ndim == 1:
        return array.reshape(-1)

    rows = array.reshape(array.shape[0], -1)
    if rows.shape[1] == 1:
        return rows[:, 0].reshape(-1)
    return _object_value_vector(tuple(row.tolist()) for row in rows)


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = _atomic_label_vector(values, name=name)
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one value per source row: {vector.shape[0]} != {expected_length}.")
    return vector


def install() -> None:
    """Patch MixStyle label handling for composite source labels."""

    source_mixstyle = importlib.import_module("neureptrace.decoding.source_mixstyle")
    original_augment = source_mixstyle.augment_source_domains_mixstyle
    if getattr(original_augment, _PATCH_MARKER, False):
        return

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
        domains = source_mixstyle._domain_vector(source_domains, expected_length=features.shape[0])
        unique_domains = tuple(dict.fromkeys(domains.tolist()))
        if len(unique_domains) < 2 and cfg.mixes_per_row > 0:
            raise ValueError("MixStyle source-domain augmentation requires at least two source domains.")
        stats = source_mixstyle._domain_stats(features, domains, unique_domains)

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
        synthetic_domain_vector = np.asarray(synthetic_domains, dtype=object)
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
    source_mixstyle.augment_source_domains_mixstyle = augment_source_domains_mixstyle


__all__ = ["install"]
