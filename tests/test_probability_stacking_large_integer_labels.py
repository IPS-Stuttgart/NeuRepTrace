from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.probability_stacking import stack_probability_observations, summarize_stacked_metrics


def _large_label_observations(*, subject: str, labels: list[int]) -> pd.DataFrame:
    first_label = 2**53
    second_label = first_label + 1
    rows: list[dict[str, object]] = []

    for decoder in ("candidate-a", "candidate-b"):
        for sample_index, true_label in enumerate(labels):
            probabilities = (0.9, 0.1) if true_label == first_label else (0.1, 0.9)
            rows.append(
                {
                    "subject": subject,
                    "fold": 0,
                    "split_id": "split-0",
                    "decoder": decoder,
                    "backend": "sklearn",
                    "emission_mode": "calibrated",
                    "time": 0.184,
                    "sample_index": sample_index,
                    "sequence_id": sample_index,
                    "true_label": true_label,
                    "true_class": f"class-{true_label}",
                    f"class_{first_label}": f"class-{first_label}",
                    f"class_{second_label}": f"class-{second_label}",
                    f"prob_class_{first_label}": probabilities[0],
                    f"prob_class_{second_label}": probabilities[1],
                }
            )

    frame = pd.DataFrame(rows)
    frame["true_label"] = pd.Series([row["true_label"] for row in rows], dtype=object)
    return frame


def test_probability_stacking_preserves_adjacent_large_integer_labels() -> None:
    first_label = 2**53
    second_label = first_label + 1
    source = _large_label_observations(
        subject="source",
        labels=[first_label, second_label, first_label, second_label],
    )
    target = _large_label_observations(
        subject="target",
        labels=[first_label, second_label],
    )

    stacked = stack_probability_observations(source, target, weighting="uniform")
    metrics = summarize_stacked_metrics(stacked)

    assert stacked["predicted_label"].tolist() == [first_label, second_label]
    assert stacked["true_label"].tolist() == [first_label, second_label]
    assert stacked["is_correct"].tolist() == [True, True]
    assert metrics["accuracy"].tolist() == pytest.approx([1.0])
    assert metrics["balanced_accuracy"].tolist() == pytest.approx([1.0])


def test_probability_stacking_rejects_large_float_label_controls() -> None:
    label = 2**53 + 2
    observations = pd.DataFrame(
        {
            "subject": ["subject-a"],
            "fold": [0],
            "decoder": ["stacked"],
            "emission_mode": ["source_oof_stacked"],
            "time": [0.184],
            "true_label": [float(label)],
            f"prob_class_{label}": [1.0],
        }
    )

    with pytest.raises(ValueError, match="exact integer representations"):
        summarize_stacked_metrics(observations)
