from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_calibration_metrics import (
    SOURCE_CALIBRATION_METRICS_CATEGORY,
    source_calibration_metrics,
)


def test_source_calibration_metrics_values_and_metadata() -> None:
    probabilities = np.asarray([[0.9, 0.1], [0.4, 0.6], [0.8, 0.2]], dtype=float)
    labels = np.asarray([0, 1, 1])

    result = source_calibration_metrics(probabilities, labels, n_bins=5)

    expected_nll = -np.mean(np.log([0.9, 0.6, 0.2]))
    assert np.isclose(result.nll, expected_nll)
    assert result.brier > 0.0
    assert 0.0 <= result.ece <= 1.0
    assert np.isclose(result.accuracy, 2.0 / 3.0)
    assert result.metadata["source_calibration_metrics_protocol_category"] == SOURCE_CALIBRATION_METRICS_CATEGORY
    assert result.metadata["source_calibration_metrics_uses_heldout_labels"] is False
    assert result.metadata["source_calibration_metrics_valid_for_strict_source_only"] is True


def test_source_calibration_metrics_normalizes_probability_rows() -> None:
    result = source_calibration_metrics([[9.0, 1.0], [1.0, 3.0]], [0, 1], n_bins="2")

    assert result.n_bins == 2
    assert result.accuracy == 1.0


def test_source_calibration_metrics_rejects_bad_labels() -> None:
    with pytest.raises(ValueError, match="outside"):
        source_calibration_metrics([[0.5, 0.5]], [2])

    with pytest.raises(ValueError, match="one value"):
        source_calibration_metrics([[0.5, 0.5]], [0, 1])


def test_source_calibration_metrics_rejects_invalid_controls() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        source_calibration_metrics([[0.5, 0.5]], [0], n_bins=0)

    with pytest.raises(ValueError, match="epsilon"):
        source_calibration_metrics([[0.5, 0.5]], [0], epsilon=1.0)
