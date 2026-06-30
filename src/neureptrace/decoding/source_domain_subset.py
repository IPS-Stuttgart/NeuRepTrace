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


def source_domain_subset_mask(
    source_domains: Sequence[Hashable] | np.ndarray,
    *,
    omit_fraction: float | str = 0.25,
    min_domains: int | str = 1,
    random_state: int | str | None = 13,
) -> SourceDomainSubsetResult:
    domains = _object_vector(source_domains, name="source_domains")
    unique_domains = tuple(dict.fromkeys(domains.tolist()))
    if not unique_domains:
        raise ValueError("At least one source domain is required.")
    min_domains = _validate_positive_int(min_domains, name="min_domains")
    omit_fraction = _validate_unit_interval(omit_fraction, name="omit_fraction")
    seed = _validate_optional_nonnegative_int(random_state, name="random_state")
    if min_domains > len(unique_domains):
        raise ValueError("min_domains is outside the valid range.")
    rng = np.random.default_rng(seed)
    n_omit = min(int(np.floor(omit_fraction * len(unique_domains))), len(unique_domains) - min_domains)
    shuffled_indices = np.arange(len(unique_domains), dtype=int)
    rng.shuffle(shuffled_indices)
    omitted = tuple(unique_domains[int(index)] for index in shuffled_indices[:n_omit])
    omitted_set = set(omitted)
    selected = tuple(domain for domain in unique_domains if domain not in omitted_set)
    mask = np.asarray([domain not in omitted_set for domain in domains.tolist()], dtype=bool)
    return SourceDomainSubsetResult(selected_mask=mask, selected_domains=selected, omitted_domains=omitted)


def apply_source_domain_subset(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[Any] | np.ndarray,
    source_domains: Sequence[Hashable] | np.ndarray,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, SourceDomainSubsetResult]:
    matrix = _feature_matrix(features, name="features")
    label_vector = _object_vector(labels, name="labels")
    if label_vector.shape[0] != matrix.shape[0]:
        raise ValueError("labels must contain one value per feature row.")
    result = source_domain_subset_mask(source_domains, **kwargs)
    if result.selected_mask.shape[0] != matrix.shape[0]:
        raise ValueError("source_domains must contain one value per feature row.")
    return matrix[result.selected_mask].astype(np.float32, copy=False), label_vector[result.selected_mask], result


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _object_vector(values: Sequence[Any] | np.ndarray, *, name: str = "values") -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a one-dimensional sequence of hashable values, not a scalar string.")
    reject_missing = name == "source_domains"
    items = _row_items(values, name=name)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        try:
            vector[index] = _hashable_value(value, reject_missing=reject_missing)
        except TypeError as exc:
            raise ValueError(f"{name} values must be hashable; got {value!r}.") from exc
        except ValueError as exc:
            raise ValueError(f"{name} values must not be missing; got {value!r}.") from exc
    return vector


def _row_items(values: Sequence[Any] | np.ndarray, *, name: str) -> list[Any]:
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            raise ValueError(f"{name} must be a one-dimensional sequence of hashable values.")
        if array.ndim == 1:
            return array.tolist()
        rows = array.reshape(array.shape[0], -1)
        if rows.shape[1] == 1:
            return rows[:, 0].tolist()
        return [tuple(row.tolist()) for row in rows]
    try:
        return list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence of hashable values.") from exc


def _missing_value(value: Any) -> bool:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        pass
    try:
        equal_to_self = value == value
    except (TypeError, ValueError):
        return False
    if isinstance(equal_to_self, np.ndarray):
        return False
    try:
        return not bool(equal_to_self)
    except (TypeError, ValueError):
        return True


def _hashable_value(value: Any, *, reject_missing: bool = False) -> Hashable:
    if isinstance(value, np.generic):
        return _hashable_value(value.item(), reject_missing=reject_missing)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _hashable_value(value.item(), reject_missing=reject_missing)
        return tuple(_hashable_value(item, reject_missing=reject_missing) for item in value.tolist())
    if isinstance(value, list):
        return tuple(_hashable_value(item, reject_missing=reject_missing) for item in value)
    if isinstance(value, tuple):
        return tuple(_hashable_value(item, reject_missing=reject_missing) for item in value)
    if reject_missing and _missing_value(value):
        raise ValueError("missing")
    hash(value)
    return value


def _validate_positive_int(value: object, *, name: str) -> int:
    scalar_value = _scalar_config_value(value, name=name, expected="a positive integer")
    if isinstance(scalar_value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        numeric = float(scalar_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0 or numeric < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(numeric)


def _validate_unit_interval(value: object, *, name: str) -> float:
    scalar_value = _scalar_config_value(value, name=name, expected="in [0, 1]")
    if isinstance(scalar_value, (bool, np.bool_)):
        raise ValueError(f"{name} must be in [0, 1].")
    try:
        numeric = float(scalar_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be in [0, 1].") from exc
    if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return numeric


def _validate_optional_nonnegative_int(value: object, *, name: str) -> int | None:
    scalar_value = _scalar_config_value(value, name=name, expected="a non-negative integer")
    if _none_like_config_value(scalar_value):
        return None
    return _validate_nonnegative_int(scalar_value, name=name)


def _none_like_config_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _scalar_config_value(value: object, *, name: str, expected: str) -> object:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be {expected}.")
        return value.item()
    if isinstance(value, (list, tuple, dict, set)):
        raise ValueError(f"{name} must be {expected}.")
    return value


def _validate_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0 or numeric < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(numeric)
