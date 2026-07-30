from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.metrics.weighted import weighted_brier_score_multiclass, weighted_top_k_accuracy


def test_weighted_probability_metrics_accept_pandas_array_like_inputs() -> None:
    probabilities = pd.DataFrame([[0.9, 0.1], [0.4, 0.6]])
    labels = pd.DataFrame({"label": [0, 1]})
    sample_weight = pd.DataFrame({"weight": [1.0, 2.0]})

    assert weighted_brier_score_multiclass(probabilities, labels, sample_weight) == pytest.approx(0.22)
    assert weighted_top_k_accuracy(probabilities, labels, sample_weight) == pytest.approx(1.0)
