import numpy as np

from neureptrace.decoding.test_time_adaptation import adapt_probabilities_online, normalize_test_time_adaptation


def test_numeric_zero_disables_test_time_adaptation_mode():
    base = np.array([[0.55, 0.45], [0.20, 0.80]])

    result = adapt_probabilities_online(base, mode="0")

    assert normalize_test_time_adaptation("0") == "none"
    assert np.allclose(result.probabilities, base)
    assert result.metadata["test_time_adaptation"] == "none"
    assert result.metadata["test_time_adaptation_uses_target_features"] is False
