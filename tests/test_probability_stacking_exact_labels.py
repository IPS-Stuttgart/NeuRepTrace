from __future__ import annotations

import pandas as pd

from neureptrace.probability_stacking import stack_probability_observations, summarize_stacked_metrics


_LABEL_A = 2**53
_LABEL_B = _LABEL_A + 1


def _observation_rows(*, subject: str, labels: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_index, true_label in enumerate(labels):
        probability_a = 0.9 if true_label == _LABEL_A else 0.1
        probability_b = 1.0 - probability_a
        rows.append(
            {
                "subject": subject,
                "fold": "fold-0",
                "decoder": "candidate",
                "sample_index": sample_index,
                "sequence_id": sample_index,
                "true_label": true_label,
                "true_class": f"class-{true_label}",
                f"class_{_LABEL_A}": "class-a",
                f"class_{_LABEL_B}": "class-b",
                f"prob_class_{_LABEL_A}": probability_a,
                f"prob_class_{_LABEL_B}": probability_b,
            }
        )
    return pd.DataFrame(rows)


def test_probability_stacking_preserves_adjacent_large_integer_labels() -> None:
    source = _observation_rows(subject="source", labels=[_LABEL_A, _LABEL_B, _LABEL_A, _LABEL_B])
    target = _observation_rows(subject="target", labels=[_LABEL_A, _LABEL_B])

    stacked = stack_probability_observations(source, target, weighting="uniform")
    metrics = summarize_stacked_metrics(stacked)

    assert stacked["predicted_label"].tolist() == [_LABEL_A, _LABEL_B]
    assert stacked["probability_true_class"].tolist() == [0.9, 0.9]
    assert stacked["is_correct"].tolist() == [True, True]
    assert metrics["accuracy"].tolist() == [1.0]
    assert metrics["balanced_accuracy"].tolist() == [1.0]
