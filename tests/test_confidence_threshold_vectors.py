from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.confidence import select_confident_rows


@pytest.mark.parametrize("parameter_name", ["min_confidence", "min_margin", "max_entropy"])
@pytest.mark.parametrize("value", [np.asarray([0.5]), np.asarray([[0.5]])])
def test_confidence_thresholds_reject_one_element_vectors(parameter_name: str, value: np.ndarray) -> None:
    with pytest.raises(ValueError, match=rf"{parameter_name} must be a scalar"):
        select_confident_rows([[0.8, 0.2]], **{parameter_name: value})


def test_confidence_thresholds_accept_zero_dimensional_arrays() -> None:
    result = select_confident_rows(
        [[0.8, 0.2]],
        min_confidence=np.asarray(0.5),
        min_margin=np.asarray(0.1),
        max_entropy=np.asarray(1.0),
    )

    assert result.accepted_mask.tolist() == [True]
