from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from neureptrace.fieldtrip_mat import load_fieldtrip_raw_mat_epochs
from neureptrace.mne_time_decode import run_time_resolved_decode


def _write_fieldtrip_raw_mat(path: Path, *, n_trials: int = 6) -> None:
    rng = np.random.default_rng(7)
    n_channels = 3
    n_times = 5
    times = np.arange(n_times, dtype=float) / 100.0
    labels = np.array([["MEG001"], ["MEG002"], ["MEG003"], ["REF001"]], dtype=object)
    trial = np.empty((1, n_trials), dtype=object)
    time = np.empty((1, n_trials), dtype=object)
    trialinfo = np.array([[1], [2]] * (n_trials // 2), dtype=int)
    for trial_index in range(n_trials):
        label = int(trialinfo[trial_index, 0]) - 1
        data = rng.normal(scale=0.05, size=(n_channels, n_times))
        data[label, :] += 1.0
        trial[0, trial_index] = data
        time[0, trial_index] = times[None, :]

    grad = {
        "label": np.array([["MEG001"], ["MEG002"], ["MEG003"]], dtype=object),
        "chantype": np.array([["meggrad"], ["meggrad"], ["meggrad"]], dtype=object),
        "coordsys": "ctf",
    }
    sio.savemat(
        path,
        {
            "data": {
                "label": labels,
                "trial": trial,
                "time": time,
                "trialinfo": trialinfo,
                "sampleinfo": np.column_stack([np.arange(n_trials) * 10, np.arange(n_trials) * 10 + n_times - 1]),
                "grad": grad,
            }
        },
    )


def test_load_fieldtrip_raw_mat_epochs_trims_overlong_labels(tmp_path: Path):
    mat_path = tmp_path / "Part10Data.mat"
    _write_fieldtrip_raw_mat(mat_path, n_trials=6)

    with pytest.warns(RuntimeWarning, match="Trimming FieldTrip data.label"):
        epochs, metadata = load_fieldtrip_raw_mat_epochs(mat_path)

    assert epochs.get_data().shape == (6, 3, 5)
    assert epochs.ch_names == ["MEG001", "MEG002", "MEG003"]
    np.testing.assert_allclose(epochs.times, np.arange(5, dtype=float) / 100.0)
    assert metadata["trialinfo"].tolist() == [1, 2, 1, 2, 1, 2]
    assert metadata["condition"].tolist() == [0, 1, 0, 1, 0, 1]
    assert metadata["sample_start"].tolist() == [0, 10, 20, 30, 40, 50]


def test_run_time_resolved_decode_accepts_fieldtrip_mat(tmp_path: Path):
    mat_path = tmp_path / "Part10Data.mat"
    _write_fieldtrip_raw_mat(mat_path, n_trials=8)

    out = tmp_path / "decode.csv"
    observations_out = tmp_path / "observations.csv"
    results = run_time_resolved_decode(
        epochs_path=mat_path,
        input_format="fieldtrip-mat",
        label_column="condition",
        out_path=out,
        n_splits=2,
        window_ms=20,
        step_ms=20,
        max_iter=2000,
        emission_mode="uncalibrated",
        observation_out_path=observations_out,
        subject="Part10",
    )

    observations = pd.read_csv(observations_out)
    assert not results.empty
    assert results["subject"].unique().tolist() == ["Part10"]
    assert observations["subject"].unique().tolist() == ["Part10"]
    assert sorted(map(str, observations["true_class"].unique())) == ["0", "1"]
