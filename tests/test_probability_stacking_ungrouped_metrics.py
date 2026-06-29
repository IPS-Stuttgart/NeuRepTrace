import pandas as pd

from neureptrace.probability_stacking import summarize_stacked_metrics


def test_summarize_stacked_metrics_accepts_ungrouped_observations():
    observations = pd.DataFrame(
        {
            "true_label": [0, 1, 1],
            "prob_class_0": [0.8, 0.3, 0.4],
            "prob_class_1": [0.2, 0.7, 0.6],
            "source_oof_candidates": ["logistic|linear_svm"] * 3,
            "source_oof_weights": ["0.4|0.6"] * 3,
            "source_oof_weighting": ["stacked"] * 3,
            "source_oof_pooling": ["linear"] * 3,
            "source_oof_balanced_accuracy": [0.75] * 3,
            "source_oof_log_loss": [0.42] * 3,
        }
    )

    summary = summarize_stacked_metrics(observations)

    assert len(summary) == 1
    assert not {"subject", "fold", "time"}.intersection(summary.columns)
    row = summary.iloc[0]
    assert row["n_test"] == 3
    assert row["n_classes"] == 2
    assert row["accuracy"] == 1.0
    assert row["balanced_accuracy"] == 1.0
    assert row["top2_accuracy"] == 1.0
    assert row["top3_accuracy"] == 1.0
    assert row["source_oof_candidates"] == "logistic|linear_svm"
    assert row["source_oof_weights"] == "0.4|0.6"
