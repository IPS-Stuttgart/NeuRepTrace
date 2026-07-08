import numpy as np
import pandas as pd
import pytest

from neureptrace.metrics import brier_score_multiclass, top_k_accuracy, validate_probability_inputs


def test_probability_metrics_accept_dataframe_probabilities_and_label_column():
    probabilities = pd.DataFrame(
        {
            "class_0": [0.8, 0.3, 0.1],
            "class_1": [0.2, 0.7, 0.9],
        }
    )
    labels = pd.DataFrame({"label": [0, 1, 1]})

    validated_probabilities, validated_labels = validate_probability_inputs(probabilities, labels)

    np.testing.assert_allclose(validated_probabilities, probabilities.to_numpy())
    np.testing.assert_array_equal(validated_labels, np.array([0, 1, 1]))
    assert top_k_accuracy(probabilities, labels, k=1) == 1.0
    assert brier_score_multiclass(probabilities, labels) == pytest.approx(np.mean([0.08, 0.18, 0.02]))
