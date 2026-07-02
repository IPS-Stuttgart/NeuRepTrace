from __future__ import annotations

import numpy as np
import pytest

import neureptrace  # noqa: F401 - importing package installs runtime compatibility patches.
from neureptrace.decoding import source_vrex


def test_source_vrex_domain_balanced_batch_uses_training_domains_without_name_error() -> None:
    train_idx = np.asarray([0, 2, 3, 5])
    domains = np.asarray(["a", "held_out", "a", "b", "held_out", "b"], dtype=object)
    rng = np.random.default_rng(17)

    batch = source_vrex._domain_balanced_batch(train_idx, domains, batch_size=6, rng=rng)

    assert batch.shape == (6,)
    assert set(batch).issubset(set(train_idx.tolist()))
    assert set(domains[batch]) == {"a", "b"}


def test_source_vrex_domain_balanced_batch_rejects_empty_training_split() -> None:
    with pytest.raises(ValueError, match="train_idx must contain at least one row"):
        source_vrex._domain_balanced_batch(np.asarray([], dtype=int), np.asarray([0, 1]), batch_size=2, rng=np.random.default_rng(3))
