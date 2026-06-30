"""Source-only domain subset generation."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from neureptrace.decoding._domain_labels import _as_domain_vector, _domain_mask, _unique_domain_labels


def domain_subsets(domains, subset_size=None):
    """Return row masks for every source-domain subset of a fixed size."""
    values = _as_domain_vector(domains)
    unique = _unique_domain_labels(values)
    if len(unique) < 2:
        raise ValueError("at least two domains are required")
    size = len(unique) - 1 if subset_size is None else _normalize_subset_size(subset_size)
    if size < 1 or size > len(unique):
        raise ValueError("subset_size is outside the valid range")
    subsets = []
    for selected in combinations(unique, size):
        mask = np.zeros(values.shape[0], dtype=bool)
        for label in selected:
            mask |= _domain_mask(values, label)
        subsets.append((selected, mask))
    return tuple(subsets)


def _scalar_subset_size_value(value, *, message: str):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(message)
    return value


def _normalize_subset_size(value) -> int:
    message = "subset_size must be a positive integer."
    scalar = _scalar_subset_size_value(value, message=message)
    try:
        parsed = float(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(message)
    return int(parsed)
