from __future__ import annotations

import pytest

from neureptrace.decoding.source_domain_subset import source_domain_subset_mask


def test_source_domain_subset_rejects_string_input() -> None:
    bad_input = "alpha"
    with pytest.raises(ValueError, match="source_domains"):
        source_domain_subset_mask(bad_input)
