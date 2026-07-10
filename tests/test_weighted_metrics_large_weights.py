from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics import validate_sample_weight as exported_validate_sample_weight
from neureptrace.metrics.weighted import (
    validate_sample_weight,
    weighted_brier_score_multiclass,
    weighted_expected_calibration_error,
    weighted_negative_log_likelihood,
    weighted_reliability_bins,
    weighted_top_k_accuracy,
)


def test_validate_sample_weight_rescales_only_when_reductions_risk_overflow() -> None:
    overflowing_total = np.asarray([1e308, 1e308])
    finite_total_large_numerator = np.asarray([1e308, 1e307])

    validated = validate_sample_weight(overflowing_total, 2)
    exported = exported_validate_sample_weight(overflowing_total, 2)

    np.testing.assert_allclose(validated, [1.0, 1.0])
    np.testing.assert_allclose(exported, validated)
    np.testing.assert_allclose(validate_sample_weight(finite_total_large_numerator, 2), [1.0, 0.1])
    np.testing.assert_allclose(validate_sample_weight([1.0, 2.0], 2), [1.0, 2.0])


def test_weighted_metrics_remain_finite_for_large_finite_weights() -> None:
    probabilities = np.asarray([[0.0, 1.0], [0.2, 0.8]])
    labels = np.asarray([0, 1])
    large = np.asarray([1e308, 1e307])
    reference = np.asarray([10.0, 1.0])

    assert weighted_brier_score_multiclass(probabilities, labels, large) == pytest.approx(
        weighted_brier_score_multiclass(probabilities, labels, reference)
    )
    assert weighted_negative_log_likelihood(probabilities, labels, large) == pytest.approx(
        weighted_negative_log_likelihood(probabilities, labels, reference)
    )
    assert weighted_top_k_accuracy(probabilities, labels, large) == pytest.approx(
        weighted_top_k_accuracy(probabilities, labels, reference)
    )
    assert weighted_expected_calibration_error(probabilities, labels, large, n_bins=2) == pytest.approx(
        weighted_expected_calibration_error(probabilities, labels, reference, n_bins=2)
    )

    rows = weighted_reliability_bins(probabilities, labels, large, n_bins=2)
    reference_rows = weighted_reliability_bins(probabilities, labels, reference, n_bins=2)
    for row, reference_row in zip(rows, reference_rows, strict=True):
        assert row["sample_weight"] == pytest.approx(reference_row["sample_weight"])
        assert row["sample_weight_fraction"] == pytest.approx(reference_row["sample_weight_fraction"])
        if reference_row["n_samples"]:
            assert row["accuracy"] == pytest.approx(reference_row["accuracy"])
            assert row["confidence"] == pytest.approx(reference_row["confidence"])
            assert row["gap"] == pytest.approx(reference_row["gap"])
