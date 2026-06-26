import numpy as np
import pytest

from neureptrace.signal.band import bandpass_sos, validate_band_hz, validate_sampling_rate, validate_signal_values


@pytest.mark.parametrize("sampling_rate", [True, False, np.bool_(True), np.bool_(False)])
def test_validate_sampling_rate_rejects_boolean_scalars(sampling_rate):
    with pytest.raises(ValueError, match="positive finite value, not boolean"):
        validate_sampling_rate(sampling_rate)


@pytest.mark.parametrize(
    "band_hz",
    [
        (True, 12.0),
        (np.bool_(True), 12.0),
        (8.0, True),
        (8.0, np.bool_(True)),
    ],
)
def test_validate_band_hz_rejects_boolean_cutoffs(band_hz):
    with pytest.raises(ValueError, match="finite numbers, not boolean"):
        validate_band_hz(band_hz, 200.0)


@pytest.mark.parametrize("order", [True, False, np.bool_(True), np.bool_(False)])
def test_bandpass_sos_rejects_boolean_filter_order(order):
    with pytest.raises(ValueError, match="positive integer, not boolean"):
        bandpass_sos(200.0, (8.0, 12.0), order=order)


@pytest.mark.parametrize("axis", [True, False, np.bool_(True), np.bool_(False)])
def test_validate_signal_values_rejects_boolean_axis(axis):
    with pytest.raises(ValueError, match="integer, not boolean"):
        validate_signal_values(np.ones((3, 4), dtype=float), axis=axis)


def test_validate_signal_values_accepts_numpy_integer_axis():
    signal = np.ones((3, 4), dtype=float)

    validated = validate_signal_values(signal, axis=np.int64(-1))

    assert validated.shape == signal.shape
