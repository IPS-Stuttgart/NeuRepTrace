import pandas as pd
import pytest

from neureptrace.probability_stacking import summarize_stacked_metrics


def test_summarize_stacked_metrics_supports_global_table_without_group_keys() -> None:
    observations = pd.DataFrame(
        {
            "true_label": [0, 1, 0],
            "prob_class_0": [0.9, 0.2, 0.8],
            "prob_class_1": [0.1, 0.8, 0.2],
            "source_oof_weights": ["0.4|0.6", "0.4|0.6", "0.4|0.6"],
        }
    )

    metrics = summarize_stacked_metrics(observations)

    assert len(metrics) == 1
    assert metrics.loc[0, "n_test"] == 3
    assert metrics.loc[0, "n_classes"] == 2
    assert metrics.loc[0, "accuracy"] == pytest.approx(1.0)
    assert metrics.loc[0, "balanced_accuracy"] == pytest.approx(1.0)
    assert metrics.loc[0, "top2_accuracy"] == pytest.approx(1.0)
    assert metrics.loc[0, "source_oof_weights"] == "0.4|0.6"


def test_summarize_stacked_metrics_checks_global_metadata_consistency() -> None:
    observations = pd.DataFrame(
        {
            "true_label": [0, 1],
            "prob_class_0": [0.9, 0.2],
            "prob_class_1": [0.1, 0.8],
            "source_oof_weights": ["1|0", "0|1"],
        }
    )

    with pytest.raises(ValueError, match="source_oof_weights"):
        summarize_stacked_metrics(observations)
