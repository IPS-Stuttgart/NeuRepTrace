from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_domain_subset import source_domain_subset_mask


@pytest.mark.parametrize(
    "domains",
    [
        ["subject_one", None, "subject_two"],
        ["subject_one", np.nan, "subject_two"],
        np.asarray([["subject_one", "session_a"], ["subject_two", np.nan]], dtype=object),
    ],
)
def test_source_domain_subset_rejects_missing_source_domain_values(domains) -> None:
    with pytest.raises(ValueError, match="missing"):
        source_domain_subset_mask(domains)
