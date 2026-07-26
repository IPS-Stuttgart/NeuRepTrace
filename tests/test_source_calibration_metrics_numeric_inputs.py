from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.decoding.source_calibration_metrics import source_calibration_metrics


@pytest.mark.parametrize(
    "probabilities",
    [
        np.asarray([[True, False]], dtype=bool),
        np.asarray([[0.5, True]], dtype=object),
    ],
)
def test_source_calibration_metrics_rejects_boolean_probabilities(probabilities: np.ndarray) -> None:
    with pytest.raises(ValueError, match="probabilities.*boolean"):
        source_calibration_metrics(probabilities, [0])


def test_source_calibration_metrics_rejects_complex_probabilities_before_real_cast() -> None:
    probabilities = np.asarray([[0.5 + 0.25j, 0.5 - 0.25j]], dtype=complex)

    with pytest.raises(ValueError, match="probabilities.*complex"):
        source_calibration_metrics(probabilities, [0])


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([True], dtype=bool),
        np.asarray([False], dtype=object),
    ],
)
def test_source_calibration_metrics_rejects_boolean_labels(labels: np.ndarray) -> None:
    with pytest.raises(ValueError, match="labels.*boolean"):
        source_calibration_metrics([[0.5, 0.5]], labels)


def test_source_calibration_metrics_rejects_complex_labels_before_real_cast() -> None:
    labels = np.asarray([0.0 + 1.0j], dtype=complex)

    with pytest.raises(ValueError, match="labels.*complex"):
        source_calibration_metrics([[0.5, 0.5]], labels)


def test_source_calibration_metrics_preserves_array_like_inputs() -> None:
    probabilities = pd.DataFrame([[0.6, 0.4], [0.2, 0.8]])
    labels = pd.Series([0, 1])

    result = source_calibration_metrics(probabilities, labels)

    assert result.accuracy == 1.0
