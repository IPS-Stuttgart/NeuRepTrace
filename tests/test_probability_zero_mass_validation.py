from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics import validate_probability_inputs


def test_validate_probability_inputs_rejects_zero_mass_rows_with_large_tolerance() -> None:
    with np.errstate(divide="raise", invalid="raise"):
        with pytest.raises(ValueError, match="probability rows must sum to one"):
            validate_probability_inputs(
                [[0.0, 0.0]],
                normalization_atol=1.0,
            )


def test_validate_probability_inputs_preserves_zero_mass_unnormalized_scores() -> None:
    scores, labels = validate_probability_inputs(
        [[0.0, 0.0]],
        require_normalized=False,
        normalization_atol=1.0,
    )

    assert labels is None
    np.testing.assert_array_equal(scores, [[0.0, 0.0]])
