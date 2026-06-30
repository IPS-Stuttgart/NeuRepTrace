import pandas as pd

from neureptrace.observation_ensemble import ensemble_probability_observations


def test_ensemble_probability_observations_handles_partial_numeric_class_columns() -> None:
    observations = pd.DataFrame(
        [
            {
                "subject": "sub-01",
                "fold": 0,
                "decoder": decoder,
                "emission_mode": "calibrated",
                "time": 0.1,
                "sample_index": 0,
                "sequence_id": 0,
                "true_label": 2,
                "true_class": "two",
                "class_0": "zero",
                "prob_class_0": 0.10,
                "prob_class_1": 0.20,
                "prob_class_2": 0.70,
            }
            for decoder in ("source_a", "source_b")
        ]
    )

    ensemble = ensemble_probability_observations(
        observations,
        decoders=("source_a", "source_b"),
        baseline_window=None,
    )

    assert ensemble["predicted_label"].tolist() == [2]
    assert ensemble["predicted_class"].tolist() == ["2"]
    assert "class_2" in ensemble.columns


def test_ensemble_probability_observations_handles_partial_named_class_columns() -> None:
    observations = pd.DataFrame(
        [
            {
                "subject": "sub-01",
                "fold": 0,
                "decoder": decoder,
                "emission_mode": "calibrated",
                "time": 0.1,
                "sample_index": 0,
                "sequence_id": 0,
                "true_label": 1,
                "true_class": "right",
                "class_left": "left",
                "prob_class_left": 0.25,
                "prob_class_right": 0.75,
            }
            for decoder in ("source_a", "source_b")
        ]
    )

    ensemble = ensemble_probability_observations(
        observations,
        decoders=("source_a", "source_b"),
        baseline_window=None,
    )

    assert ensemble["predicted_label"].tolist() == [1]
    assert ensemble["predicted_class"].tolist() == ["right"]
    assert "class_right" in ensemble.columns


def test_ensemble_probability_observations_fills_rowwise_missing_class_values() -> None:
    observations = pd.DataFrame(
        [
            {
                "subject": "sub-01",
                "fold": 0,
                "decoder": "source_a",
                "emission_mode": "calibrated",
                "time": 0.1,
                "sample_index": 0,
                "sequence_id": 0,
                "true_label": 1,
                "true_class": "one",
                "class_0": "zero",
                "class_1": "one",
                "prob_class_0": 0.20,
                "prob_class_1": 0.80,
            },
            {
                "subject": "sub-01",
                "fold": 0,
                "decoder": "source_b",
                "emission_mode": "calibrated",
                "time": 0.1,
                "sample_index": 0,
                "sequence_id": 0,
                "true_label": 1,
                "true_class": "one",
                "prob_class_0": 0.30,
                "prob_class_1": 0.70,
            },
        ]
    )

    ensemble = ensemble_probability_observations(
        observations,
        decoders=("source_a", "source_b"),
        baseline_window=None,
    )

    assert ensemble["predicted_label"].tolist() == [1]
    assert ensemble["predicted_class"].tolist() == ["one"]
