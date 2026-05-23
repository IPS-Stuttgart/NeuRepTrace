from __future__ import annotations

import numpy as np

import neureptrace  # noqa: F401
from neureptrace.dataset_spec import _first_time_axis


def test_first_time_axis_preserves_matlab_column_vector() -> None:
    times = np.array([[-0.1], [0.0], [0.1]])

    assert np.allclose(_first_time_axis(times), [-0.1, 0.0, 0.1])


def test_first_time_axis_preserves_matlab_row_vector() -> None:
    times = np.array([[-0.1, 0.0, 0.1]])

    assert np.allclose(_first_time_axis(times), [-0.1, 0.0, 0.1])


def test_first_time_axis_keeps_first_row_for_time_matrix() -> None:
    times = np.array([[-0.1, 0.0, 0.1], [-0.2, 0.0, 0.2]])

    assert np.allclose(_first_time_axis(times), [-0.1, 0.0, 0.1])
