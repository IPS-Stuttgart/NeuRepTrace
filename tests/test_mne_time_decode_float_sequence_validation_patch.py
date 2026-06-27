from __future__ import annotations

import numpy as np
import pytest

from neureptrace.mne_time_decode import _parse_float_sequence


@pytest.mark.parametrize(
    "bad_times",
    [
        "0.088,nan",
        "0.088 inf",
        [0.088, np.nan],
        [0.088, np.inf],
        np.array([0.088, np.nan]),
        [True, 0.184],
        True,
    ],
)
def test_mne_time_decode_float_sequences_reject_nonfinite_or_boolean_values(bad_times) -> None:
    with pytest.raises(ValueError, match="decode_candidate_times must contain finite numeric time values"):
        _parse_float_sequence(bad_times, default=(0.088,))


def test_mne_time_decode_float_sequence_rejects_nonfinite_defaults() -> None:
    with pytest.raises(ValueError, match="decode_candidate_times must contain finite numeric time values"):
        _parse_float_sequence(None, default=(0.088, float("inf")))


def test_mne_time_decode_float_sequences_accept_numpy_arrays() -> None:
    assert _parse_float_sequence(np.array([0.088, 0.184]), default=()) == (0.088, 0.184)
