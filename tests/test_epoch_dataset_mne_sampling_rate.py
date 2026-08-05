from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.io.dataset import EpochDataset


def _epoch_dataset(times: list[float]) -> EpochDataset:
    return EpochDataset(
        data=np.ones((1, 1, len(times)), dtype=float),
        times=np.asarray(times, dtype=float),
        channel_names=["MEG001"],
        metadata=pd.DataFrame({"condition": ["a"]}),
    )


def test_to_mne_epochs_rejects_sampling_rate_that_changes_time_axis() -> None:
    dataset = _epoch_dataset([0.0, 0.1, 0.2])

    with pytest.raises(ValueError, match="must match"):
        dataset.to_mne_epochs(sfreq=20.0)


def test_to_mne_epochs_rejects_irregular_times_even_with_explicit_sampling_rate() -> None:
    dataset = _epoch_dataset([0.0, 0.1, 0.25])

    with pytest.raises(ValueError, match="uniformly sampled"):
        dataset.to_mne_epochs(sfreq=10.0)


@pytest.mark.parametrize(
    "sfreq",
    [
        True,
        np.bool_(False),
        np.complex128(10.0 + 1.0j),
        np.asarray(10.0),
        0.0,
        float("inf"),
    ],
)
def test_to_mne_epochs_rejects_invalid_explicit_sampling_rates(sfreq: object) -> None:
    dataset = _epoch_dataset([0.0, 0.1, 0.2])

    with pytest.raises(ValueError, match="positive finite scalar"):
        dataset.to_mne_epochs(sfreq=sfreq)


def test_to_mne_epochs_preserves_time_axis_for_matching_numpy_scalar_rate() -> None:
    dataset = _epoch_dataset([0.0, 0.1, 0.2])

    epochs = dataset.to_mne_epochs(sfreq=np.float64(10.0))

    assert epochs.info["sfreq"] == pytest.approx(10.0)
    assert np.allclose(epochs.times, dataset.times, rtol=0.0, atol=1e-12)
