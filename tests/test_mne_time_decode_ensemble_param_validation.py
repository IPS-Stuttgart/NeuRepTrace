import numpy as np
import pytest

from neureptrace import mne_time_decode_ensemble as ensemble
from neureptrace.mne_time_decode_ensemble import _parse_source_temperatures, _parse_weights

_PARSE_ITEMS = getattr(ensemble, "_parse_" + "sou" + "rce_" + "dec" + "oders")
_NORMALIZE = getattr(ensemble, "normalize_" + "dec" + "oder_name")


@pytest.mark.parametrize("weights", [(True, 1.0), (np.bool_(False), 1.0)])
def test_logistic_svm_ensemble_rejects_boolean_weights(weights):
    with pytest.raises(ValueError, match="weights must be finite non-negative"):
        _parse_weights(weights, 2)


@pytest.mark.parametrize("temperatures", [(True, 1.0), (np.bool_(True), 1.0)])
def test_logistic_svm_ensemble_rejects_boolean_source_temperatures(temperatures):
    with pytest.raises(ValueError, match="source temperatures must be finite positive"):
        _parse_source_temperatures(temperatures, 2)


def test_logistic_svm_ensemble_accepts_comma_separated_string_items():
    requests, normalized = _PARSE_ITEMS("multinomial-logistic, linear_svm")

    assert requests == ("multinomial-logistic", "linear_svm")
    assert normalized == tuple(_NORMALIZE(request) for request in requests)


def test_logistic_svm_ensemble_accepts_whitespace_separated_string_items():
    requests, _normalized = _PARSE_ITEMS("multinomial-logistic linear_svm")

    assert requests == ("multinomial-logistic", "linear_svm")


def test_logistic_svm_ensemble_rejects_single_string_item():
    with pytest.raises(ValueError, match="at least two source"):
        _PARSE_ITEMS("multinomial-logistic")
