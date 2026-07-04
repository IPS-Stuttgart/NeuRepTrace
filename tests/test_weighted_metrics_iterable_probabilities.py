from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics.weighted import (
    weighted_brier_score_multiclass,
    weighted_expected_calibration_error,
    weighted_negative_log_likelihood,
    weighted_reliability_bins,
    weighted_top_k_accuracy,
)


def _probability_rows():
    yield [0.9, 0.1]
    yield [0.4, 0.6]
    yield [0.8, 0.2]


def _label_indices():
    yield 0
    yield 1
    yield 1


def test_weighted_probability_metrics_accept_one_pass_probability_iterables() -> None:
    labels = np.asarray([0, 1, 1])
    weights = np.asarray([1.0, 2.0, 3.0])

    assert weighted_brier_score_multiclass(_probability_rows(), labels, weights) == pytest.approx(0.75)
    assert weighted_negative_log_likelihood(_probability_rows(), labels, weights) == pytest.approx(
        -np.average(np.log([0.9, 0.6, 0.2]), weights=weights)
    )
    assert weighted_top_k_accuracy(_probability_rows(), labels, weights, k=1) == pytest.approx(0.5)
    assert weighted_expected_calibration_error(_probability_rows(), labels, weights, n_bins=2) == pytest.approx(0.25)

    rows = weighted_reliability_bins(_probability_rows(), labels, weights, n_bins=2)
    assert rows[1]["sample_weight"] == pytest.approx(6.0)
    assert rows[1]["accuracy"] == pytest.approx(0.5)


def test_weighted_probability_metrics_accept_one_pass_label_iterables() -> None:
    weights = np.asarray([1.0, 2.0, 3.0])

    assert weighted_top_k_accuracy(_probability_rows(), _label_indices(), weights, k=1) == pytest.approx(0.5)


def test_weighted_probability_iterables_still_reject_boolean_values() -> None:
    labels = np.asarray([0, 1])
    weights = np.asarray([1.0, 1.0])
    probability_rows = (row for row in ([True, False], [False, True]))

    with pytest.raises(ValueError, match="probabilities.*boolean"):
        weighted_top_k_accuracy(probability_rows, labels, weights)
