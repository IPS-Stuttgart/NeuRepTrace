import numpy as np
import pytest

from neureptrace.signal.band import bandpass_sos, validate_band_hz, validate_sampling_rate, validate_signal_values


@pytest.mark.parametrize("axis", [False, True, np.bool_(True)])
def test_validate_signal_values_rejects_boolean_axis(axis):
    signal = np.ones((3, 4), dtype=float)

    with pytest.raises(ValueError, match="axis must be an integer"):
        validate_signal_values(signal, axis=axis)


def test_validate_signal_values_accepts_numpy_integer_axis():
    signal = np.ones((3, 4), dtype=float)

    validated = validate_signal_values(signal, axis=np.int64(-1))

    assert validated.shape == signal.shape


@pytest.mark.parametrize("order", [True, np.bool_(True)])
def test_bandpass_sos_rejects_boolean_filter_order(order):
    with pytest.raises(ValueError, match="filter order must be a positive integer"):
        bandpass_sos(100.0, (8.0, 12.0), order=order)


@pytest.mark.parametrize("sampling_rate", [True, np.bool_(True)])
def test_validate_sampling_rate_rejects_boolean_values(sampling_rate):
    with pytest.raises(ValueError, match="sampling_rate must be a positive finite value"):
        validate_sampling_rate(sampling_rate)


@pytest.mark.parametrize("band_hz", [(True, 12.0), (8.0, True), (np.bool_(True), 12.0), (8.0, np.bool_(True))])
def test_validate_band_hz_rejects_boolean_cutoffs(band_hz):
    with pytest.raises(ValueError, match="Cutoff frequencies must be finite numbers"):
        validate_band_hz(band_hz, 100.0)
