from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.confidence_filter import confidence_filter, probability_entropy


def test_confidence_filter_basic_thresholds() -> None:
    rows = np.asarray([[0.90, 0.10], [0.55, 0.45], [0.10, 0.80]], dtype=float)
    result = confidence_filter(rows, min_confidence=0.70, min_margin=0.20)
    assert result.predicted_index.tolist() == [0, 0, 1]
    assert result.accepted_mask.tolist() == [True, False, True]
    assert result.metadata["confidence_filter_n_accepted"] == 2


def test_entropy_and_row_normalization() -> None:
    rows = np.asarray([[5.0, 5.0], [10.0, 0.0]], dtype=float)
    entropy = probability_entropy(rows)
    result = confidence_filter(rows, min_confidence=0.8)
    assert np.allclose(entropy, [1.0, 0.0], atol=1e-6)
    assert np.allclose(result.confidence, [0.5, 1.0])
    assert result.accepted_mask.tolist() == [False, True]


def test_entropy_normalization_accepts_string_boolean_config() -> None:
    rows = np.asarray([[0.5, 0.5], [1.0, 0.0]], dtype=float)

    entropy = probability_entropy(rows, normalize="false")
    result = confidence_filter(rows, max_entropy=0.8, normalize_entropy="false")

    assert np.allclose(entropy, [np.log(2.0), 0.0], atol=1e-6)
    assert result.accepted_mask.tolist() == [True, True]
    assert result.metadata["confidence_filter_entropy_normalized"] is False


@pytest.mark.parametrize("value", [None, "", " none ", "NULL", np.asarray("none")])
def test_confidence_filter_accepts_none_like_max_entropy(value: object) -> None:
    rows = np.asarray([[0.5, 0.5], [1.0, 0.0]], dtype=float)

    result = confidence_filter(rows, max_entropy=value)

    assert result.metadata["confidence_filter_max_entropy"] == ""


@pytest.mark.parametrize("value", [0.8, "0.8", np.asarray(0.8)])
def test_confidence_filter_accepts_scalar_max_entropy(value: object) -> None:
    rows = np.asarray([[0.5, 0.5], [1.0, 0.0]], dtype=float)

    result = confidence_filter(rows, max_entropy=value)

    assert result.metadata["confidence_filter_max_entropy"] == 0.8


@pytest.mark.parametrize("value", [-0.1, "bad", [0.5], {"threshold": 0.5}, np.asarray([0.5, 0.6])])
def test_confidence_filter_rejects_invalid_max_entropy(value: object) -> None:
    rows = np.asarray([[0.5, 0.5], [1.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="max_entropy"):
        confidence_filter(rows, max_entropy=value)


def test_entropy_normalization_rejects_invalid_boolean_config() -> None:
    rows = np.asarray([[0.5, 0.5], [1.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="normalize"):
        probability_entropy(rows, normalize="sometimes")
    with pytest.raises(ValueError, match="normalize_entropy"):
        confidence_filter(rows, normalize_entropy="sometimes")
