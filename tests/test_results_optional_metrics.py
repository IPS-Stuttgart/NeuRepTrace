import pandas as pd
import pytest

from neureptrace.results import aggregate_time_decode_results, subject_time_metrics


def test_aggregate_time_decode_results_preserves_optional_accuracy_metrics() -> None:
    results = pd.DataFrame(
        {
            "subject": ["s1", "s1", "s2", "s2"],
            "fold": [0, 1, 0, 1],
            "time": [0.1, 0.1, 0.1, 0.1],
            "accuracy": [0.1, 0.3, 0.5, 0.7],
            "log_loss": [1.0, 0.8, 0.6, 0.4],
            "brier": [0.9, 0.7, 0.5, 0.3],
            "ece": [0.2, 0.4, 0.6, 0.8],
            "balanced_accuracy": [0.2, 0.6, 0.4, 0.8],
            "top2_accuracy": [1.0, 1.0, 0.0, 1.0],
            "top3_accuracy": [1.0, 1.0, 1.0, 1.0],
            "n_test": [1, 3, 2, 2],
        }
    )

    subject_time = subject_time_metrics(results)
    aggregated = aggregate_time_decode_results(results)

    assert subject_time["balanced_accuracy"].tolist() == pytest.approx([0.5, 0.6])
    assert subject_time["top2_accuracy"].tolist() == pytest.approx([1.0, 0.5])
    assert "balanced_accuracy_mean" in aggregated.columns
    assert "top2_accuracy_mean" in aggregated.columns
    assert "top3_accuracy_mean" in aggregated.columns
    assert aggregated["balanced_accuracy_mean"].tolist() == pytest.approx([0.55])
    assert aggregated["top2_accuracy_mean"].tolist() == pytest.approx([0.75])
    assert aggregated["top3_accuracy_mean"].tolist() == pytest.approx([1.0])
