import numpy as np
import pytest

from neureptrace.decoding.test_time_adaptation import (
    TTIME_PROTOCOL,
    adapt_probabilities_online,
    normalize_test_time_adaptation,
    normalize_ttime_update_timing,
)


def test_ttime_after_predict_is_online_future_only():
    base = np.array(
        [
            [0.80, 0.20],
            [0.70, 0.30],
            [0.30, 0.70],
        ]
    )

    result = adapt_probabilities_online(
        base,
        source_prior=[0.5, 0.5],
        learning_rate=0.5,
        entropy_weight=1.0,
        marginal_weight=0.0,
        marginal_momentum=1.0,
        update_timing="after_predict",
    )

    assert result.probabilities.shape == base.shape
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert np.allclose(result.probabilities[0], base[0])
    assert not np.allclose(result.final_bias, 0.0)
    assert result.metadata["test_time_adaptation_protocol"] == TTIME_PROTOCOL
    assert result.metadata["test_time_adaptation_uses_target_features"] is True
    assert result.metadata["test_time_adaptation_uses_target_labels"] is False
    assert result.metadata["test_time_adaptation_update_timing"] == "after_predict"


def test_ttime_before_predict_can_adapt_current_row():
    base = np.array([[0.90, 0.10]])

    result = adapt_probabilities_online(
        base,
        source_prior=[0.5, 0.5],
        learning_rate=0.5,
        entropy_weight=1.0,
        marginal_weight=0.0,
        marginal_momentum=1.0,
        update_timing="before_predict",
    )

    assert result.probabilities[0, 0] > base[0, 0]
    assert result.probabilities[0, 1] < base[0, 1]


def test_ttime_none_returns_input_probabilities_with_disabled_metadata():
    base = np.array([[0.55, 0.45], [0.20, 0.80]])

    result = adapt_probabilities_online(base, mode="none", source_prior=[2, 1])

    assert np.allclose(result.probabilities, base)
    assert result.metadata["test_time_adaptation"] == "none"
    assert result.metadata["test_time_adaptation_uses_target_features"] is False
    assert result.metadata["test_time_adaptation_uses_target_labels"] is False
    assert np.allclose(result.running_marginal, [2 / 3, 1 / 3])


def test_ttime_alias_normalization():
    assert normalize_test_time_adaptation(None) == "none"
    assert normalize_test_time_adaptation("t-time") == "ttime"
    assert normalize_test_time_adaptation("online-entropy") == "ttime"
    assert normalize_ttime_update_timing("future-only") == "after_predict"
    assert normalize_ttime_update_timing("adapt-current") == "before_predict"


def test_ttime_rejects_invalid_probabilities_and_priors():
    with pytest.raises(ValueError, match="sum to 1.0"):
        adapt_probabilities_online([[0.5, 0.6]])

    with pytest.raises(ValueError, match="source_prior"):
        adapt_probabilities_online([[0.5, 0.5]], source_prior=[1.0, 0.0, 0.0])
