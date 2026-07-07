from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.io.dataset import EpochDataset


def _epoch_dataset(times: list[float]) -> EpochDataset:
    return EpochDataset(
        data=np.ones((1, 1, len(times))),
        times=np.asarray(times, dtype=float),
        channel_names=["MEG001"],
        metadata=pd.DataFrame({"condition": ["a"]}),
    )


def test_epoch_dataset_infers_sampling_frequency_from_uniform_time_axis() -> None:
    assert _epoch_dataset([0.0, 0.1, 0.2]).infer_sampling_frequency() == pytest.approx(10.0)


def test_epoch_dataset_rejects_irregular_time_axis_for_sampling_frequency() -> None:
    dataset = _epoch_dataset([0.0, 0.1, 0.25])

    with pytest.raises(ValueError, match="uniformly sampled"):
        dataset.infer_sampling_frequency()


def test_epoch_dataset_rejects_nonfinite_time_axis() -> None:
    with pytest.raises(ValueError, match="finite"):
        _epoch_dataset([0.0, float("nan")])
