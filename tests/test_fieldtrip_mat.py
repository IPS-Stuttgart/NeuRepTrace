from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from neureptrace.fieldtrip_mat import _sampleinfo_array, load_fieldtrip_raw_mat_epochs, write_fieldtrip_raw_mat_epochs


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


def test_write_fieldtrip_mat_refuses_existing_metadata_without_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class DummyEpochs:
        def __init__(self) -> None:
            self.saved_paths: list[tuple[Path, bool]] = []

        def save(self, path: Path, *, overwrite: bool) -> None:
            self.saved_paths.append((path, overwrite))

    epochs = DummyEpochs()
    metadata = pd.DataFrame({"trial": [0], "condition": [1]})

    def fake_loader(mat_path: Path | str, **kwargs):
        return epochs, metadata

    monkeypatch.setattr("neureptrace.fieldtrip_mat.load_fieldtrip_raw_mat_epochs", fake_loader)
    metadata_out = tmp_path / "already_there.csv"
    metadata_out.write_text("old\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Metadata output already exists"):
        write_fieldtrip_raw_mat_epochs(
            tmp_path / "dummy.mat",
            epochs_out=tmp_path / "converted-epo.fif",
            metadata_out=metadata_out,
        )

    assert epochs.saved_paths == []
    assert metadata_out.read_text(encoding="utf-8") == "old\n"


def test_fieldtrip_sampleinfo_accepts_integral_float_bounds():
    sampleinfo = _sampleinfo_array(np.array([[1.0, 5.0], [6.0, 10.0]]), n_trials=2)

    assert sampleinfo.tolist() == [[1, 5], [6, 10]]
    assert np.issubdtype(sampleinfo.dtype, np.integer)


@pytest.mark.parametrize(
    "sampleinfo",
    [
        np.array([[1.5, 5.0], [6.0, 10.0]], dtype=float),
        np.array([[True, False], [False, True]], dtype=bool),
        np.array([[1, 5], [10, 6]], dtype=int),
    ],
)
def test_fieldtrip_sampleinfo_rejects_malformed_bounds(sampleinfo: np.ndarray):
    with pytest.raises(ValueError, match="sampleinfo must contain finite integer sample bounds"):
        _sampleinfo_array(sampleinfo, n_trials=2)
