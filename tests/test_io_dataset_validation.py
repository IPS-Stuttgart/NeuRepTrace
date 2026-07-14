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


def test_epoch_dataset_rejects_duplicate_channel_names():
    with pytest.raises(ValueError, match="channel_names.*unique"):
        _dataset(channel_names=["MEG001", "MEG001"])


def test_with_channels_rejects_duplicate_requested_channels():
    dataset = _dataset()

    with pytest.raises(ValueError, match="Requested channel names.*unique"):
        dataset.with_channels(["MEG001", "MEG001"])
