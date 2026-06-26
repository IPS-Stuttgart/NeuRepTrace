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
    domains = _object_vector(source_domains, name="source_domains")
    unique_domains = tuple(dict.fromkeys(domains.tolist()))
    if not unique_domains:
        raise ValueError("At least one source domain is required.")
    min_domains = _validate_positive_int(min_domains, name="min_domains")
    omit_fraction = _validate_unit_interval(omit_fraction, name="omit_fraction")
    if min_domains > len(unique_domains):
        raise ValueError("min_domains is outside the valid range.")
    rng = np.random.default_rng(random_state)
    n_omit = min(int(np.floor(omit_fraction * len(unique_domains))), len(unique_domains) - min_domains)
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
    label_vector = _object_vector(labels, name="labels")
    if label_vector.shape[0] != matrix.shape[0]:
        raise ValueError("labels must contain one value per feature row.")
    result = source_domain_subset_mask(source_domains, **kwargs)
    if result.selected_mask.shape[0] != matrix.shape[0]:
        raise ValueError("source_domains must contain one value per feature row.")
    return matrix[result.selected_mask].astype(np.float32, copy=False), label_vector[result.selected_mask], result


def _object_vector(values: Sequence[Any] | np.ndarray, *, name: str = "values") -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a one-dimensional sequence of hashable values, not a scalar string.")
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence of hashable values.") from exc
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"{name} values must be hashable; got {value!r}.") from exc
        vector[index] = value
    return vector


def _validate_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0 or numeric < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(numeric)


def _validate_unit_interval(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be in [0, 1].")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be in [0, 1].") from exc
    if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return numeric
