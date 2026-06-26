import numpy as np
import pytest

from neureptrace.metrics import brier_score_multiclass, validate_probability_inputs


def test_probability_metric_labels_reject_boolean_dtype() -> None:
    probabilities = np.array([[0.7, 0.3], [0.2, 0.8]])
    labels = np.array([True, False])

    with pytest.raises(ValueError, match="labels must contain integer class indices"):
        validate_probability_inputs(probabilities, labels)


@pytest.mark.parametrize(
    "labels",
    [
        np.array([0, False], dtype=object),
        np.array([np.int64(0), np.bool_(True)], dtype=object),
    ],
)
def test_probability_metrics_reject_embedded_boolean_labels(labels: np.ndarray) -> None:
    probabilities = np.array([[0.7, 0.3], [0.2, 0.8]])

    with pytest.raises(ValueError, match="labels must contain integer class indices"):
        brier_score_multiclass(probabilities, labels)


def test_probability_metric_labels_keep_accepting_integer_like_values() -> None:
    probabilities = np.array([[0.7, 0.3], [0.2, 0.8]])
    labels = np.array(["0", 1.0], dtype=object)

    _, coerced = validate_probability_inputs(probabilities, labels)

    np.testing.assert_array_equal(coerced, np.array([0, 1]))
