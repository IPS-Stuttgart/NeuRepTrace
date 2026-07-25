import numpy as np
import pytest

from neureptrace.signal.band import validate_band_hz


@pytest.mark.parametrize(
    "band_hz",
    [
        "12",
        b"12",
        bytearray(b"12"),
        np.str_("12"),
        np.bytes_("12"),
    ],
)
def test_validate_band_hz_rejects_complete_textual_values(band_hz):
    with pytest.raises(ValueError, match="exactly two cutoff frequencies"):
        validate_band_hz(band_hz, 200.0)


def test_validate_band_hz_still_accepts_two_numeric_cutoffs():
    assert validate_band_hz((8.0, 12.0), 200.0) == (8.0, 12.0)
