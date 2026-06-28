from __future__ import annotations

import pytest

import neureptrace  # noqa: F401
from neureptrace.decoding import (
    normalize_decoder_name,
    normalize_emission_mode,
    normalize_feature_preprocessor,
    normalize_tuning_scoring,
)


@pytest.mark.parametrize(
    ("normalizer", "value", "message"),
    [
        (normalize_decoder_name, True, "decoder must be a string"),
        (normalize_decoder_name, None, "decoder must be a string"),
        (normalize_emission_mode, 1, "emission_mode must be a string"),
        (normalize_feature_preprocessor, True, "feature_preprocessor must be a string or None"),
        (normalize_tuning_scoring, ["accuracy"], "tuning_scoring must be a string"),
    ],
)
def test_decoding_option_normalizers_reject_non_string_values(normalizer, value, message):
    with pytest.raises(ValueError, match=message):
        normalizer(value)


def test_feature_preprocessor_none_still_normalizes_to_identity():
    assert normalize_feature_preprocessor(None) == "none"
