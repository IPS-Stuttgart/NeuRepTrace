from __future__ import annotations

import numpy as np
import pytest

from neureptrace.fieldtrip_mat import _sampling_properties


@pytest.mark.parametrize(
    "times",
    [
        np.asarray([[0.0, np.inf]], dtype=float),
        np.asarray([[0.0, np.nextafter(0.0, 1.0)]], dtype=float),
    ],
    ids=["non-finite-axis", "reciprocal-overflow"],
)
def test_fieldtrip_sampling_properties_reject_invalid_derived_rates(times: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite positive sampling frequency"):
        _sampling_properties(times)


def test_fieldtrip_sampling_properties_preserve_regular_time_axes() -> None:
    sampling_rate, tmin = _sampling_properties(np.asarray([[-0.1, 0.0, 0.1]], dtype=float))

    assert sampling_rate == pytest.approx(10.0)
    assert tmin == pytest.approx(-0.1)
