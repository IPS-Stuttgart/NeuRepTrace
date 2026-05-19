from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from neureptrace.mne_transfer_decode import run_time_resolved_transfer_decode


def _write_epochs(path: Path, *, offset: float = 0.0, labels: list[str] | None = None) -> None:
    if labels is None:
        labels = ["a", "b"] * 8
    n_epochs = len(labels)
    n_times = 12
    sfreq = 100.0
    info = mne.create_info(["MEG001", "MEG002"], sfreq=sfreq, ch_types=["grad", "grad"])
    data = np.zeros((n_epochs, 2, n_times), dtype=float)
    for index, label in enumerate(labels):
        sign = -1.0 if label == "a" else 1.0
        data[index, 0, :] = sign + offset
        data[index, 1, :] = 0.1 * sign
    events = np.column_stack([np.arange(n_epochs), np.zeros(n_epochs, dtype=int), np.arange(n_epochs) % 2 + 1])
    metadata = pd.DataFrame({"condition": labels, "session": ["transfer-test"] * n_epochs})
    epochs = mne.EpochsArray(data, info, events=events, tmin=0.0, metadata=metadata, verbose="error")
    epochs.save(path, overwrite=True)


def test_time_resolved_transfer_decode_trains_on_one_epochs_file_and_tests_on_another(tmp_path: Path):
    train_path = tmp_path / "train-epo.fif"
    test_path = tmp_path / "test-epo.fif"
    out_path = tmp_path / "transfer.csv"
    observations_path = tmp_path / "observations.csv"
    _write_epochs(train_path)
    _write_epochs(test_path, offset=0.05)

    results = run_time_resolved_transfer_decode(
        train_path,
        test_path,
        label_column="condition",
        out_path=out_path,
        window_ms=40.0,
        step_ms=40.0,
        decoder="logistic",
        emission_mode="uncalibrated",
        max_iter=200,
        train_recording="main",
        test_recording="cue",
        train_subject="Part10",
        test_subject="Part10",
        observation_out_path=observations_path,
    )

    assert out_path.exists()
    assert observations_path.exists()
    assert set(results["split_kind"]) == {"transfer"}
    assert set(results["train_recording"]) == {"main"}
    assert set(results["test_recording"]) == {"cue"}
    assert results["accuracy"].min() > 0.9
    observations = pd.read_csv(observations_path)
    assert {"train_recording", "test_recording", "probability_true_class"}.issubset(observations.columns)


def test_time_resolved_transfer_decode_rejects_unseen_test_classes(tmp_path: Path):
    train_path = tmp_path / "train-epo.fif"
    test_path = tmp_path / "test-epo.fif"
    _write_epochs(train_path, labels=["a", "b"] * 4)
    _write_epochs(test_path, labels=["a", "c"] * 4)

    with pytest.raises(ValueError, match="absent from the transfer training set"):
        run_time_resolved_transfer_decode(
            train_path,
            test_path,
            label_column="condition",
            out_path=tmp_path / "transfer.csv",
            window_ms=40.0,
            step_ms=40.0,
            decoder="logistic",
            emission_mode="uncalibrated",
            max_iter=200,
        )
