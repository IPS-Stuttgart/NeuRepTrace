"""Source-only domain subset generation."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def domain_subsets(domains, subset_size=None):
    """Return row masks for every source-domain subset of a fixed size."""
    values = np.asarray(domains, dtype=object).reshape(-1)
    unique = tuple(dict.fromkeys(values.tolist()))
    if len(unique) < 2:
        raise ValueError("at least two domains are required")
    size = len(unique) - 1 if subset_size is None else int(subset_size)
    if size < 1 or size > len(unique):
        raise ValueError("subset_size is outside the valid range")
    return tuple(
        (selected, np.asarray([value in set(selected) for value in values], dtype=bool))
        for selected in combinations(unique, size)
    )
