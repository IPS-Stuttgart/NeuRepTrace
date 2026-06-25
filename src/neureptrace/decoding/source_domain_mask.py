"""Strict source-only source-domain masking utilities."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_DOMAIN_MASK_PROTOCOL = "strict_source_only_domain_mask"
SOURCE_DOMAIN_MASK_CATEGORY = "1_strict_source_only"


@dataclass(frozen=True, slots=True)
class SourceDomainMaskResult:
    """Selected rows/domains and provenance for source-domain masking."""

    selected_mask: np.ndarray
    selected_domains: tuple[Hashable, ...]
    heldout_domains: tuple[Hashable, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def source_domain_mask(
    source_domains: Sequence[Hashable] | np.ndarray,
    *,
    holdout_fraction: float | str = 0.25,
    min_selected_domains: int | str = 1,
    random_state: int | str | None = 13,
) -> SourceDomainMaskResult:
    """Select a deterministic subset of source domains.

    This helper uses source-domain ids only and is valid for strict source-only
    ablation or source-ensemble experiments.  It does not accept target features
    or target labels.
    """

    domains = _object_vector(source_domains, name="source_domains")
    unique_domains = tuple(dict.fromkeys(domains.tolist()))
    if not unique_domains:
        raise ValueError("At least one source domain is required.")
    min_keep = _positive_int(min_selected_domains, name="min_selected_domains")
    if min_keep > len(unique_domains):
        raise ValueError("min_selected_domains cannot exceed the number of source domains.")
    fraction = _unit_interval(holdout_fraction, name="holdout_fraction")
    seed = None if random_state in {None, "", "none", "None"} else _nonnegative_int(random_state, name="random_state")
    n_holdout = min(int(np.floor(fraction * len(unique_domains))), len(unique_domains) - min_keep)
    shuffled = np.asarray(unique_domains, dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    heldout_domains = tuple(shuffled[:n_holdout].tolist())
    heldout_set = set(heldout_domains)
    selected_domains = tuple(domain for domain in unique_domains if domain not in heldout_set)
    selected_mask = np.asarray([domain not in heldout_set for domain in domains.tolist()], dtype=bool)
    metadata = {
        "source_domain_mask": True,
        "source_domain_mask_protocol": SOURCE_DOMAIN_MASK_PROTOCOL,
        "source_domain_mask_protocol_category": SOURCE_DOMAIN_MASK_CATEGORY,
        "source_domain_mask_uses_source_domains": True,
        "source_domain_mask_uses_target_features": False,
        "source_domain_mask_uses_target_labels": False,
        "source_domain_mask_valid_for_strict_source_only": True,
        "source_domain_mask_n_rows": int(domains.shape[0]),
        "source_domain_mask_n_domains": int(len(unique_domains)),
        "source_domain_mask_n_selected_domains": int(len(selected_domains)),
        "source_domain_mask_n_heldout_domains": int(len(heldout_domains)),
        "source_domain_mask_holdout_fraction": float(fraction),
        "source_domain_mask_min_selected_domains": int(min_keep),
        "source_domain_mask_random_state": "" if seed is None else int(seed),
        "source_domain_mask_selected_domains": "|".join(str(domain) for domain in selected_domains),
        "source_domain_mask_heldout_domains": "|".join(str(domain) for domain in heldout_domains),
    }
    return SourceDomainMaskResult(
        selected_mask=selected_mask,
        selected_domains=selected_domains,
        heldout_domains=heldout_domains,
        metadata=metadata,
    )


def apply_source_domain_mask(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[Any] | np.ndarray,
    source_domains: Sequence[Hashable] | np.ndarray,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, SourceDomainMaskResult]:
    """Return feature/label rows selected by :func:`source_domain_mask`."""

    matrix = _feature_matrix(features, name="features")
    label_vector = _object_vector(labels, name="labels")
    if label_vector.shape[0] != matrix.shape[0]:
        raise ValueError("labels must contain one value per feature row.")
    result = source_domain_mask(source_domains, **kwargs)
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


def _object_vector(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
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


def _positive_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _nonnegative_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(parsed)


def _unit_interval(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be in [0, 1].")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed
