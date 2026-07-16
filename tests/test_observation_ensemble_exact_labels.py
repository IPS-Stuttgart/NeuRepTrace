from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.observation_ensemble import (
    ensemble_probability_observations,
    summarize_ensemble_metrics,
)


@pytest.mark.parametrize(
    "labels",
    [
        (-1, 2),
        (2**53, 2**53 + 1),
    ],
)
def test_observation_ensemble_preserves_exact_probability_labels(labels: tuple[int, int]) -> None:
    label_a, label_b = labels
    rows: list[dict[str, object]] = []
    for decoder in ("source_a", "source_b"):
        for sample_index, true_label in enumerate(labels):
            probability_a = 0.9 if true_label == label_a else 0.1
            rows.append(
                {
                    "subject": "sub-01",
                    "fold": 0,
                    "decoder": decoder,
                    "emission_mode": "calibrated",
                    "time": 0.1,
                    "sample_index": sample_index,
                    "sequence_id": sample_index,
                    "true_label": true_label,
                    "true_class": f"class-{true_label}",
                    f"class_{label_a}": f"class-{label_a}",
                    f"class_{label_b}": f"class-{label_b}",
                    f"prob_class_{label_a}": probability_a,
                    f"prob_class_{label_b}": 1.0 - probability_a,
                }
            )

    ensemble = ensemble_probability_observations(
        pd.DataFrame(rows),
        decoders=("source_a", "source_b"),
        baseline_window=None,
    )
    metrics = summarize_ensemble_metrics(ensemble)

    assert ensemble["predicted_label"].tolist() == [label_a, label_b]
    np.testing.assert_allclose(ensemble["probability_true_class"], 0.9)
    assert ensemble["is_correct"].tolist() == [True, True]
    assert metrics["accuracy"].tolist() == [1.0]
    assert metrics["balanced_accuracy"].tolist() == [1.0]
