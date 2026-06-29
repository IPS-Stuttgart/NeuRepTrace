import numpy as np

import neureptrace  # noqa: F401 - installs runtime compatibility patches
import neureptrace.io.fieldtrip_mat as fieldtrip_mat


def _assert_repeated_time_vector(time_field):
    times = fieldtrip_mat._normalize_times(time_field, n_trials=2)

    assert len(times) == 2
    np.testing.assert_allclose(times[0], [0.0, 0.01, 0.02])
    np.testing.assert_allclose(times[1], [0.0, 0.01, 0.02])
    assert times[0] is not times[1]


def test_fieldtrip_row_time_vector_is_shared_across_trials():
    _assert_repeated_time_vector(np.array([[0.0, 0.01, 0.02]]))


def test_fieldtrip_column_time_vector_is_shared_across_trials():
    _assert_repeated_time_vector(np.array([[0.0], [0.01], [0.02]]))
