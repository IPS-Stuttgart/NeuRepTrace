"""Helpers for deterministic source-domain subsetting."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SourceDomainSubsetResult:
    selected_mask: np.ndarray
    selected_domains: tuple[Hashable, ...]
    omitted_domains: tuple[Hashable, ...]


def source_domain_subset_mask(source_domains: Sequence[Hashable] | np.ndarray, *, omit_fraction: float = 0.25, min_domains: int = 1, random_state: int | None = 13) -> SourceDomainSubsetResult:
    domains = _object_vector(source_domains)
    unique_domains = tuple(dict.fromkeys(domains.tolist()))
    if not unique_domains:
        raise ValueError("At least one source domain is required.")
    if min_domains < 1 or min_domains > len(unique_domains):
        raise ValueError("min_domains is outside the valid range.")
    if not 0.0 <= float(omit_fraction) <= 1.0:
        raise ValueError("omit_fraction must be in [0, 1].")
    rng = np.random.default_rng(random_state)
    n_omit = min(int(np.floor(float(omit_fraction) * len(unique_domains))), len(unique_domains) - int(min_domains))
    shuffled = np.asarray(unique_domains, dtype=object)
    rng.shuffle(shuffled)
    omitted = tuple(shuffled[:n_omit].tolist())
    omitted_set = set(omitted)
    selected = tuple(domain for domain in unique_domains if domain not in omitted_set)
    mask = np.asarray([domain not in omitted_set for domain in domains.tolist()], dtype=bool)
    return SourceDomainSubsetResult(selected_mask=mask, selected_domains=selected, omitted_domains=omitted)


def apply_source_domain_subset(features: Sequence[Sequence[float]] | np.ndarray, labels: Sequence[Any] | np.ndarray, source_domains: Sequence[Hashable] | np.ndarray, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, SourceDomainSubsetResult]:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix.")
    label_vector = _object_vector(labels)
    if label_vector.shape[0] != matrix.shape[0]:
        raise ValueError("labels must contain one value per feature row.")
    result = source_domain_subset_mask(source_domains, **kwargs)
    if result.selected_mask.shape[0] != matrix.shape[0]:
        raise ValueError("source_domains must contain one value per feature row.")
    return matrix[result.selected_mask].astype(np.float32, copy=False), label_vector[result.selected_mask], result


def _object_vector(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        hash(value)
        vector[index] = value
    return vector
