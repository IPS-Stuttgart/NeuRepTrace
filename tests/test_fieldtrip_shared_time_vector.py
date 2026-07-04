from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

import neureptrace  # noqa: F401 - installs runtime compatibility patches
import neureptrace.io.fieldtrip_mat as fieldtrip_mat
from neureptrace.fieldtrip_mat import load_fieldtrip_raw_mat


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


def _cell_row(values):
    cell = np.empty((1, len(values)), dtype=object)
    for index, value in enumerate(values):
        cell[0, index] = value
    return cell


def _cell_column(values):
    cell = np.empty((len(values), 1), dtype=object)
    for index, value in enumerate(values):
        cell[index, 0] = value
    return cell


def test_raw_fieldtrip_loader_reuses_single_shared_time_vector(tmp_path: Path):
    n_trials = 4
    n_channels = 2
    n_times = 5
    shared_time = np.linspace(-0.2, 0.2, n_times)[None, :]
    trials = [np.full((n_channels, n_times), float(index)) for index in range(n_trials)]
    data = {
        "label": _cell_column(["MEG001", "MEG002"]),
        "trial": _cell_row(trials),
        "time": shared_time,
        "trialinfo": np.arange(1, n_trials + 1, dtype=int)[:, None],
        "sampleinfo": np.column_stack(
            [
                np.arange(1, n_trials * n_times + 1, n_times),
                np.arange(n_times, (n_trials + 1) * n_times, n_times),
            ]
        ),
    }
    mat_path = tmp_path / "shared_time.mat"
    sio.savemat(mat_path, {"data": data})

    loaded = load_fieldtrip_raw_mat(mat_path)

    assert loaded.trials.shape == (n_trials, n_channels, n_times)
    assert loaded.times.shape == (n_trials, n_times)
    np.testing.assert_allclose(loaded.times, np.repeat(shared_time, n_trials, axis=0))
    assert loaded.sfreq == 10.0
