import pandas as pd

from neureptrace.results import subject_time_metrics


def test_observations_inherit_singleton_nondefault_condition():
    results = pd.DataFrame(
        {
            "subject": ["s1", "s1"],
            "time": [0.1, 0.1],
            "accuracy": [0.6, 0.8],
            "log_loss": [0.5, 0.4],
            "brier": [0.3, 0.2],
            "ece": [0.9, 0.9],
            "emission_mode": ["uncalibrated", "uncalibrated"],
        }
    )
    observations = pd.DataFrame(
        {
            "subject": ["s1", "s1"],
            "time": [0.1, 0.1],
            "true_label": [0, 1],
            "prob_class_0": [0.8, 0.2],
            "prob_class_1": [0.2, 0.8],
        }
    )

    metrics = subject_time_metrics(results, observations=observations)

    assert metrics["emission_mode"].tolist() == ["uncalibrated"]
    assert metrics["ece"].round(6).tolist() == [0.2]
