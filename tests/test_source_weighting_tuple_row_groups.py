from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_weighting import sample_weights_from_group_weights


def test_sample_weights_preserve_tuple_row_groups() -> None:
    row_groups = [("s1", "run1"), ("s1", "run1"), ("s2", "run2")]

    sample_weights = sample_weights_from_group_weights(
        row_groups,
        {("s1", "run1"): 2.0, ("s2", "run2"): 0.5},
        normalize=False,
    )

    assert sample_weights.shape == (3,)
    assert sample_weights.tolist() == pytest.approx([2.0, 2.0, 0.5])


def test_sample_weights_preserve_matrix_row_groups() -> None:
    row_groups = np.asarray([["s1", "run1"], ["s2", "run2"], ["s3", "run3"]], dtype=object)

    sample_weights = sample_weights_from_group_weights(
        row_groups,
        {("s1", "run1"): 2.0, ("s2", "run2"): 0.5},
        default=0.25,
        normalize=False,
    )

    assert sample_weights.shape == (3,)
    assert sample_weights.tolist() == pytest.approx([2.0, 0.5, 0.25])
