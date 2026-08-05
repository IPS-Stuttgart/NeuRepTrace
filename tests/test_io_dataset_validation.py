from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.io.dataset import EpochDataset


def _dataset(**overrides):
    defaults = {
        "data": np.ones((2, 2, 3), dtype=float),
        "times": np.array([0.0, 0.1, 0.2], dtype=float),
        "channel_names": ["MEG001", "MEG002"],
        "metadata": pd.DataFrame({"trial": [0, 1]}),
    }
    defaults.update(overrides)
    return EpochDataset(**defaults)


def test_epoch_dataset_rejects_boolean_data_values():
    data = np.array([[[True, False, True]]], dtype=bool)

    with pytest.raises(ValueError, match="data.*boolean"):
        _dataset(data=data, channel_names=["MEG001"], metadata=pd.DataFrame({"trial": [0]}))


def test_epoch_dataset_rejects_boolean_time_values():
    with pytest.raises(ValueError, match="times.*boolean"):
        _dataset(times=np.array([False, True, True], dtype=bool))


def test_epoch_dataset_rejects_complex_data_values():
    data = np.ones((2, 2, 3), dtype=np.complex128)
    data[0, 0, 0] += 1.0j

    with pytest.raises(ValueError, match="data.*complex"):
        _dataset(data=data)


def test_epoch_dataset_rejects_complex_time_values():
    times = np.array([0.0, 0.1 + 0.01j, 0.2], dtype=np.complex128)

    with pytest.raises(ValueError, match="times.*complex"):
        _dataset(times=times)


def test_epoch_dataset_rejects_duplicate_channel_names():
    with pytest.raises(ValueError, match="channel_names.*unique"):
        _dataset(channel_names=["MEG001", "MEG001"])


def test_with_channels_rejects_duplicate_requested_channels():
    dataset = _dataset()

    with pytest.raises(ValueError, match="Requested channel names.*unique"):
        dataset.with_channels(["MEG001", "MEG001"])


def test_with_channels_snapshots_selected_channel_provenance():
    dataset = _dataset()
    requested_channels = ["MEG002", "MEG001"]

    selected = dataset.with_channels(requested_channels)
    requested_channels[:] = ["MEG001"]

    assert selected.channel_names == ["MEG002", "MEG001"]
    assert selected.provenance["selected_channels"] == ["MEG002", "MEG001"]
