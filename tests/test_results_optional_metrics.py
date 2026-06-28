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


def test_aggregate_time_decode_results_allows_partially_populated_optional_metrics() -> None:
    results = pd.DataFrame(
        {
            "subject": ["s1", "s1", "s1", "s1"],
            "fold": [0, 1, 0, 1],
            "time": [0.1, 0.1, 0.1, 0.1],
            "emission_mode": [
                "calibrated",
                "calibrated",
                "calibrated_temporal_posterior",
                "calibrated_temporal_posterior",
            ],
            "accuracy": [0.4, 0.6, 0.8, 0.9],
            "log_loss": [1.0, 0.8, 0.5, 0.4],
            "brier": [0.7, 0.5, 0.3, 0.2],
            "ece": [0.3, 0.2, 0.15, 0.1],
            "balanced_accuracy": [None, None, 0.7, 0.9],
            "top2_accuracy": [None, None, 1.0, 1.0],
            "top3_accuracy": [None, None, 1.0, 1.0],
            "n_test": [2, 2, 1, 3],
        }
    )

    subject_time = subject_time_metrics(results)
    aggregated = aggregate_time_decode_results(results)

    base_subject_row = subject_time.loc[subject_time["emission_mode"] == "calibrated"].iloc[0]
    smoothed_subject_row = subject_time.loc[subject_time["emission_mode"] == "calibrated_temporal_posterior"].iloc[0]
    base_row = aggregated.loc[aggregated["emission_mode"] == "calibrated"].iloc[0]
    smoothed_row = aggregated.loc[aggregated["emission_mode"] == "calibrated_temporal_posterior"].iloc[0]

    assert pd.isna(base_subject_row["balanced_accuracy"])
    assert smoothed_subject_row["balanced_accuracy"] == pytest.approx(0.85)
    assert "balanced_accuracy_mean" in aggregated.columns
    assert pd.isna(base_row["balanced_accuracy_mean"])
    assert smoothed_row["balanced_accuracy_mean"] == pytest.approx(0.85)
    assert smoothed_row["top2_accuracy_mean"] == pytest.approx(1.0)
    assert smoothed_row["top3_accuracy_mean"] == pytest.approx(1.0)
