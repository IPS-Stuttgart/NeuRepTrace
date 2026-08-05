from __future__ import annotations

import numpy as np

import neureptrace  # noqa: F401 - installs runtime compatibility patches
import neureptrace.io.fieldtrip_mat as fieldtrip_mat


def test_fieldtrip_time_matrix_uses_columns_when_second_axis_matches_trials() -> None:
    time_matrix = np.asarray(
        [
            [0.00, 0.10],
            [0.01, 0.11],
            [0.02, 0.12],
        ]
    )

    times = fieldtrip_mat._normalize_times(time_matrix, n_trials=2)

    assert len(times) == 2
    np.testing.assert_allclose(times[0], [0.00, 0.01, 0.02])
    np.testing.assert_allclose(times[1], [0.10, 0.11, 0.12])


def test_fieldtrip_time_matrix_preserves_row_oriented_trial_axis() -> None:
    time_matrix = np.asarray(
        [
            [0.00, 0.01, 0.02],
            [0.10, 0.11, 0.12],
        ]
    )

    times = fieldtrip_mat._normalize_times(time_matrix, n_trials=2)

    assert len(times) == 2
    np.testing.assert_allclose(times[0], [0.00, 0.01, 0.02])
    np.testing.assert_allclose(times[1], [0.10, 0.11, 0.12])
