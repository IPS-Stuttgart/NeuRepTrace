from __future__ import annotations

import numpy as np
import pytest

from neureptrace.features.oscillatory import compute_band_features, compute_band_trial_features


def _time_axis(n_samples: int = 64) -> np.ndarray:
    return np.arange(n_samples, dtype=float) / 200.0


@pytest.mark.parametrize(
    "signal",
    [
        np.ones((2, 64), dtype=bool),
        np.asarray([[True] * 64, [False] * 64], dtype=object),
    ],
)
def test_compute_band_trial_features_rejects_boolean_signal_values(signal) -> None:
    with pytest.raises(ValueError, match="boolean"):
        compute_band_trial_features(signal, _time_axis())


@pytest.mark.parametrize(
    "data",
    [
        np.ones((2, 3, 64), dtype=bool),
        np.asarray(np.ones((2, 3, 64), dtype=bool), dtype=object),
    ],
)
def test_compute_band_features_rejects_boolean_data_values(data) -> None:
    with pytest.raises(ValueError, match="boolean"):
        compute_band_features(data, _time_axis(), windows=[(0.05, 0.2)])
