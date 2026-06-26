import pandas as pd

from neureptrace.observation_ensemble import summarize_ensemble_metrics


def test_summarize_ensemble_metrics_handles_ungrouped_observations() -> None:
    observations = pd.DataFrame(
        {
            "true_label": [0, 1, 1],
            "class_0": ["zero", "zero", "zero"],
            "class_1": ["one", "one", "one"],
            "prob_class_0": [0.8, 0.3, 0.2],
            "prob_class_1": [0.2, 0.7, 0.8],
        }
    )

    metrics = summarize_ensemble_metrics(observations)

    assert len(metrics) == 1
    assert "time" not in metrics.columns
    assert metrics["accuracy"].tolist() == [1.0]
    assert metrics["balanced_accuracy"].tolist() == [1.0]
    assert metrics["n_test"].tolist() == [3]
    assert metrics["n_classes"].tolist() == [2]
    assert metrics["class_names"].tolist() == ["zero|one"]
