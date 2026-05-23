from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics import weighted_reliability_bins as exported_weighted_reliability_bins
from neureptrace.metrics.weighted import (
    validate_sample_weight,
    weighted_brier_score_multiclass,
    weighted_expected_calibration_error,
    weighted_negative_log_likelihood,
    weighted_reliability_bins,
    weighted_top_k_accuracy,
)


def test_weighted_probability_metrics_match_manual_averages() -> None:
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.4, 0.6],
            [0.8, 0.2],
        ]
    )
    labels = np.array([0, 1, 1])
    sample_weight = np.array([1.0, 2.0, 3.0])

    assert weighted_brier_score_multiclass(probabilities, labels, sample_weight) == pytest.approx(0.75)
    assert weighted_negative_log_likelihood(probabilities, labels, sample_weight) == pytest.approx(
        -np.average(np.log([0.9, 0.6, 0.2]), weights=sample_weight)
    )
    assert weighted_top_k_accuracy(probabilities, labels, sample_weight, k=1) == pytest.approx(0.5)
    assert weighted_top_k_accuracy(probabilities, labels, sample_weight, k=2) == 1.0
    assert weighted_expected_calibration_error(probabilities, labels, sample_weight, n_bins=2) == pytest.approx(0.25)


def test_weighted_reliability_bins_report_weighted_calibration_rows() -> None:
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.4, 0.6],
            [0.8, 0.2],
            [0.55, 0.45],
        ]
    )
    labels = np.array([0, 1, 1, 0])
    sample_weight = np.array([1.0, 2.0, 3.0, 0.0])

    rows = weighted_reliability_bins(probabilities, labels, sample_weight, n_bins=2)

    assert rows[0] == {
        "bin": 0,
        "bin_left": 0.0,
        "bin_right": 0.5,
        "n_samples": 0,
        "sample_weight": 0.0,
        "sample_weight_fraction": 0.0,
        "accuracy": pytest.approx(float("nan"), nan_ok=True),
        "confidence": pytest.approx(float("nan"), nan_ok=True),
        "gap": pytest.approx(float("nan"), nan_ok=True),
    }
    assert rows[1]["bin"] == 1
    assert rows[1]["n_samples"] == 4
    assert rows[1]["sample_weight"] == pytest.approx(6.0)
    assert rows[1]["sample_weight_fraction"] == pytest.approx(1.0)
    assert rows[1]["accuracy"] == pytest.approx(0.5)
    assert rows[1]["confidence"] == pytest.approx(np.average([0.9, 0.6, 0.8], weights=[1.0, 2.0, 3.0]))
    assert rows[1]["gap"] == pytest.approx(rows[1]["accuracy"] - rows[1]["confidence"])


def test_weighted_reliability_bins_are_available_from_public_metrics_api() -> None:
    probabilities = np.array([[0.6, 0.4], [0.2, 0.8]])
    labels = np.array([0, 1])
    sample_weight = np.array([1.0, 1.0])

    assert exported_weighted_reliability_bins(probabilities, labels, sample_weight, n_bins=2) == weighted_reliability_bins(
        probabilities,
        labels,
        sample_weight,
        n_bins=2,
    )


def test_validate_sample_weight_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="same samples"):
        validate_sample_weight(np.array([1.0]), 2)

    with pytest.raises(ValueError, match="non-negative"):
        validate_sample_weight(np.array([1.0, -1.0]), 2)

    with pytest.raises(ValueError, match="positive total"):
        validate_sample_weight(np.array([0.0, 0.0]), 2)

    with pytest.raises(ValueError, match="finite"):
        validate_sample_weight(np.array([1.0, np.nan]), 2)


def test_weighted_probability_metrics_validate_inputs() -> None:
    probabilities = np.array([[0.7, 0.3], [0.4, 0.6]])
    labels = np.array([0, 1])
    sample_weight = np.array([1.0, 1.0])

    with pytest.raises(ValueError, match="sum to one"):
        weighted_brier_score_multiclass(np.array([[0.7, 0.7], [0.4, 0.6]]), labels, sample_weight)

    with pytest.raises(ValueError, match="valid column indices"):
        weighted_negative_log_likelihood(probabilities, np.array([0, 2]), sample_weight)

    with pytest.raises(ValueError, match="positive"):
        weighted_top_k_accuracy(probabilities, labels, sample_weight, k=0)

    with pytest.raises(ValueError, match="positive"):
        weighted_reliability_bins(probabilities, labels, sample_weight, n_bins=0)
