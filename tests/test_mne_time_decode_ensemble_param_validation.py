import numpy as np
import pytest

from neureptrace.mne_time_decode_ensemble import _parse_source_temperatures, _parse_weights


@pytest.mark.parametrize("weights", [(True, 1.0), (np.bool_(False), 1.0)])
def test_logistic_svm_ensemble_rejects_boolean_weights(weights):
    with pytest.raises(ValueError, match="weights must be finite non-negative"):
        _parse_weights(weights, 2)


@pytest.mark.parametrize("temperatures", [(True, 1.0), (np.bool_(True), 1.0)])
def test_logistic_svm_ensemble_rejects_boolean_source_temperatures(temperatures):
    with pytest.raises(ValueError, match="source temperatures must be finite positive"):
        _parse_source_temperatures(temperatures, 2)
