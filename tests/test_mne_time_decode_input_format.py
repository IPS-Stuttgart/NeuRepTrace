from __future__ import annotations

import pytest

from neureptrace.mne_time_decode import _parse_path_tokens, normalize_input_format


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ("data", 0)),
        ("data,0", ("data", 0)),
        ("outer,data,0", ("outer", "data", 0)),
        (["data", "0"], ("data", 0)),
        (("data", 0), ("data", 0)),
    ],
)
def test_parse_path_tokens(raw, expected):
    assert _parse_path_tokens(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "mne-epochs"),
        ("mne", "mne-epochs"),
        ("fif", "mne-epochs"),
        ("fieldtrip", "fieldtrip-mat"),
        ("fieldtrip_raw_mat", "fieldtrip-mat"),
        ("mat", "fieldtrip-mat"),
    ],
)
def test_normalize_input_format_aliases(raw, expected):
    assert normalize_input_format(raw) == expected
