import pandas as pd

from neureptrace.temporal_smoothing import metrics_from_probability_observations


def test_temporal_smoothing_topk_ties_use_exact_k_class_order() -> None:
    observations = pd.DataFrame(
        [
            {
                "subject": "sub-01",
                "time": 0.1,
                "true_label": 0,
                "prob_class_0": 1.0 / 3.0,
                "prob_class_1": 1.0 / 3.0,
                "prob_class_2": 1.0 / 3.0,
            },
            {
                "subject": "sub-01",
                "time": 0.1,
                "true_label": 0,
                "prob_class_0": 1.0 / 3.0,
                "prob_class_1": 1.0 / 3.0,
                "prob_class_2": 1.0 / 3.0,
            },
        ]
    )

    metrics = metrics_from_probability_observations(observations)

    row = metrics.iloc[0]
    assert row["top2_accuracy"] == 1.0
    assert row["top3_accuracy"] == 1.0


def test_temporal_smoothing_topk_ties_use_probability_column_labels() -> None:
    observations = pd.DataFrame(
        [
            {
                "subject": "sub-01",
                "time": 0.1,
                "true_label": 10,
                "prob_class_10": 1.0 / 3.0,
                "prob_class_20": 1.0 / 3.0,
                "prob_class_30": 1.0 / 3.0,
            },
            {
                "subject": "sub-01",
                "time": 0.1,
                "true_label": 10,
                "prob_class_10": 1.0 / 3.0,
                "prob_class_20": 1.0 / 3.0,
                "prob_class_30": 1.0 / 3.0,
            },
        ]
    )

    metrics = metrics_from_probability_observations(observations)

    row = metrics.iloc[0]
    assert row["top2_accuracy"] == 1.0
    assert row["top3_accuracy"] == 1.0
