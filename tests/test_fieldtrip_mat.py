from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from neureptrace.fieldtrip_mat import load_fieldtrip_raw_mat_epochs


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


def _write_fieldtrip_raw_mat(path: Path, *, cell_builder=_cell_row) -> None:
    n_trials = 6
    n_channels = 3
    n_times = 5
    time = np.linspace(-0.5, 0.5, n_times)[None, :]
    trials = [np.full((n_channels, n_times), float(index)) for index in range(n_trials)]
    labels = _cell_column(["MEG001", "MEG002", "MEG003", "REF001", "STATUS"])
    grad = {
        "label": labels,
        "chantype": _cell_column(["meggrad", "meggrad", "meggrad", "refmag", "status"]),
        "chanunit": _cell_column(["T/m", "T/m", "T/m", "T", ""]),
        "chanpos": np.zeros((5, 3), dtype=float),
        "coordsys": "ctf",
    }
    data = {
        "label": labels,
        "trial": cell_builder(trials),
        "time": cell_builder([time for _ in range(n_trials)]),
        "trialinfo": np.array([[1], [2], [3], [1], [2], [3]], dtype=int),
        "sampleinfo": np.array([[1, 5], [6, 10], [11, 15], [16, 20], [21, 25], [26, 30]], dtype=int),
        "grad": grad,
    }
    sio.savemat(path, {"data": data})


def test_load_fieldtrip_mat_trims_overlong_channel_metadata(tmp_path: Path):
    mat_path = tmp_path / "Part10Data.mat"
    _write_fieldtrip_raw_mat(mat_path)

    with pytest.warns(RuntimeWarning) as warning_record:
        epochs, metadata = load_fieldtrip_raw_mat_epochs(mat_path)

    warning_text = "\n".join(str(warning.message) for warning in warning_record)
    assert "Trimming FieldTrip data.label" in warning_text
    assert "grad.label" in warning_text
    assert "grad.chantype" in warning_text
    assert "grad.chanunit" in warning_text
    assert "grad.chanpos" in warning_text
    assert epochs.get_data(copy=False).shape == (6, 3, 5)
    assert epochs.ch_names == ["MEG001", "MEG002", "MEG003"]
    assert metadata["condition"].tolist() == [0, 1, 2, 0, 1, 2]
    assert metadata["sample_start"].tolist() == [1, 6, 11, 16, 21, 26]


def test_load_fieldtrip_mat_accepts_column_cell_arrays(tmp_path: Path):
    mat_path = tmp_path / "Part10Data.mat"
    _write_fieldtrip_raw_mat(mat_path, cell_builder=_cell_column)

    with pytest.warns(RuntimeWarning):
        epochs, metadata = load_fieldtrip_raw_mat_epochs(mat_path)

    assert epochs.get_data(copy=False).shape == (6, 3, 5)
    pd.testing.assert_series_equal(metadata["trialinfo"], pd.Series([1, 2, 3, 1, 2, 3], name="trialinfo"))


def test_load_fieldtrip_mat_can_fail_on_overlong_labels(tmp_path: Path):
    mat_path = tmp_path / "Part10Data.mat"
    _write_fieldtrip_raw_mat(mat_path)

    with pytest.raises(ValueError, match="data.label"):
        load_fieldtrip_raw_mat_epochs(mat_path, trim_overlong_labels=False)
