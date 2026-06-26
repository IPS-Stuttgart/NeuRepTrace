"""Source-only domain subset generation."""

from __future__ import annotations

from itertools import combinations

from neureptrace.decoding._domain_ids import atomic_domain_vector, domain_mask, ordered_unique


def domain_subsets(domains, subset_size=None):
    """Return row masks for every source-domain subset of a fixed size."""
    values = atomic_domain_vector(domains)
    unique = ordered_unique(values)
    if len(unique) < 2:
        raise ValueError("at least two domains are required")
    size = len(unique) - 1 if subset_size is None else int(subset_size)
    if size < 1 or size > len(unique):
        raise ValueError("subset_size is outside the valid range")
    rows = []
    for selected_domains in combinations(unique, size):
        rows.append((selected_domains, domain_mask(values, selected_domains)))
    return tuple(rows)
