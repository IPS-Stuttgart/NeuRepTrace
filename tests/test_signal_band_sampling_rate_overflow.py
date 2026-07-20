from __future__ import annotations

import numpy as np
import pytest

from neureptrace.signal import sampling_rate_from_time_axis as exported_sampling_rate
from neureptrace.signal.band import sampling_rate_from_time_axis, sampling_rate_from_time_vector


@pytest.mark.parametrize(
    "operation",
    [sampling_rate_from_time_axis, sampling_rate_from_time_vector, exported_sampling_rate],
)
def test_sampling_rate_helpers_reject_reciprocal_overflow(operation) -> None:
    smallest_interval = np.nextafter(0.0, 1.0)
    time_vector = np.asarray([0.0, smallest_interval], dtype=float)

    with pytest.raises(ValueError, match="finite sampling rate"):
        operation(time_vector)


def test_sampling_rate_guard_preserves_ordinary_axes() -> None:
    assert sampling_rate_from_time_axis([0.0, 0.01, 0.02]) == pytest.approx(100.0)
