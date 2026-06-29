from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_domain_mask import source_domain_mask


def test_source_domain_mask_accepts_none_like_seed_values() -> None:
    for seed in [None, "", " none ", "NULL", np.asarray("none")]:
        result = source_domain_mask(["a", "a", "b", "b", "c", "c"], random_state=seed)
        assert result.metadata["source_domain_mask_random_state"] == ""


def test_source_domain_mask_accepts_scalar_array_seed_values() -> None:
    result = source_domain_mask(["a", "a", "b", "b", "c", "c"], random_state=np.asarray(7))
    assert result.metadata["source_domain_mask_random_state"] == 7


def test_source_domain_mask_rejects_invalid_seed_values() -> None:
    for seed in [True, -1, 1.5, [1], np.asarray([1])]:
        with pytest.raises(ValueError, match="random_state"):
            source_domain_mask(["a", "a", "b", "b"], random_state=seed)
