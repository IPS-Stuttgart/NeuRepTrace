import numpy as np
import pandas as pd
import pytest

from neureptrace.probability_stacking import (
    _top_k_accuracy,
    _top_k_accuracy_from_label_values,
    summarize_stacked_metrics,
)


def test_probability_stacking_top_k_ties_prefer_lower_column_indices() -> None:
    probabilities = np.array(
        [
            [0.50, 0.50, 0.00],
            [0.20, 0.40, 0.40],
        ]
    )
    labels = np.array([0, 1])

    assert _top_k_accuracy(probabilities, labels, k=1) == pytest.approx(1.0)


def test_probability_stacking_top_k_label_values_respects_stable_ties() -> None:
    probabilities = np.array(
        [
            [0.50, 0.25, 0.25],
            [0.10, 0.45, 0.45],
        ]
    )
    true_labels = np.array([20, 30])
    label_values = (10, 20, 30)

    assert _top_k_accuracy_from_label_values(probabilities, true_labels, label_values, k=2) == pytest.approx(1.0)


def test_stacked_metrics_top2_uses_stable_tie_rule_for_label_values() -> None:
    observations = pd.DataFrame(
        {
            "subject": ["s1", "s1", "s1"],
            "fold": [0, 0, 0],
            "decoder": ["stacked", "stacked", "stacked"],
            "emission_mode": ["source_oof_stacked", "source_oof_stacked", "source_oof_stacked"],
            "time": [0.184, 0.184, 0.184],
            "window_start": [0.16, 0.16, 0.16],
            "window_stop": [0.20, 0.20, 0.20],
            "true_label": [20, 10, 30],
            "prob_class_10": [0.50, 0.50, 0.10],
            "prob_class_20": [0.25, 0.25, 0.45],
            "prob_class_30": [0.25, 0.25, 0.45],
        }
    )

    metrics = summarize_stacked_metrics(observations)

    assert metrics["top2_accuracy"].tolist() == pytest.approx([1.0])
    assert metrics["top3_accuracy"].tolist() == pytest.approx([1.0])
