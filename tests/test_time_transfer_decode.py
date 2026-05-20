from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from neureptrace.time_transfer_decode import run_time_transfer_decode


def _write_epochs(path: Path, *, offset: float = 0.0, n_epochs: int = 8) -> None:
    labels = np.asarray([0, 1] * (n_epochs // 2), dtype=int)
    data = np.zeros((n_epochs, 2, 21), dtype=float)
    for index, label in enumerate(labels):
        data[index, 0, 8:13] = float(label) + offset
        data[index, 1, 8:13] = float(1 - label) - offset
    info = mne.create_info(["MEG001", "MEG002"], sfreq=100.0, ch_types="grad")
    events = np.column_stack([np.arange(n_epochs), np.zeros(n_epochs, dtype=int), labels + 1])
    epochs = mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"0": 1, "1": 2},
        tmin=-0.1,
        metadata=pd.DataFrame({"condition": labels}),
        verbose="error",
    )
    epochs.save(path, overwrite=True)


def test_time_transfer_decode_writes_metrics_and_observations(tmp_path: Path):
    train_path = tmp_path / "train-epo.fif"
    validation_path = tmp_path / "validation-epo.fif"
    out_path = tmp_path / "transfer.csv"
    observations_path = tmp_path / "observations.csv"
    _write_epochs(train_path, offset=0.0, n_epochs=8)
    _write_epochs(validation_path, offset=0.05, n_epochs=8)

    results = run_time_transfer_decode(
        train_epochs_path=train_path,
        validation_epochs_path=validation_path,
        label_column="condition",
        out_path=out_path,
        window_ms=50.0,
        step_ms=50.0,
        decoder="gaussian_nb",
        emission_mode="uncalibrated",
        observation_out_path=observations_path,
        train_subject="main",
        validation_subject="cue",
        transfer_label="main-to-cue",
    )

    assert out_path.exists()
    assert observations_path.exists()
    assert not results.empty
    assert set(results["transfer"]) == {"main-to-cue"}
    assert set(results["train_subject"]) == {"main"}
    assert set(results["validation_subject"]) == {"cue"}
    assert results["n_train"].eq(8).all()
    assert results["n_test"].eq(8).all()
    observations = pd.read_csv(observations_path)
    assert not observations.empty
    assert "prob_class_0" in observations.columns
    assert "prob_class_1" in observations.columns


def test_time_transfer_decode_train_window_ensemble(tmp_path: Path):
    train_path = tmp_path / "train-epo.fif"
    validation_path = tmp_path / "validation-epo.fif"
    out_path = tmp_path / "transfer.csv"
    _write_epochs(train_path, offset=0.0, n_epochs=8)
    _write_epochs(validation_path, offset=0.05, n_epochs=8)

    results = run_time_transfer_decode(
        train_epochs_path=train_path,
        validation_epochs_path=validation_path,
        label_column="condition",
        out_path=out_path,
        window_ms=50.0,
        step_ms=50.0,
        decoder="gaussian_nb",
        emission_mode="uncalibrated",
        temporal_train_window=(-0.025, 0.025),
        transfer_label="main-to-cue",
    )

    assert set(results["temporal_mode"]) == {"train_window_transfer_ensemble"}
    assert results["n_train_windows"].ge(1).all()
