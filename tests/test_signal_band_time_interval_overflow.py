import numpy as np
import pytest

from neureptrace.signal.band import sampling_rate_from_time_axis, uniform_sample_interval, validate_time_axis


@pytest.mark.parametrize(
    "operation",
    [validate_time_axis, uniform_sample_interval, sampling_rate_from_time_axis],
)
def test_time_axis_helpers_reject_overflowing_finite_intervals(operation):
    time_vector = np.array([-1.0e308, 1.0e308], dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="sample intervals must be finite"):
            operation(time_vector)
