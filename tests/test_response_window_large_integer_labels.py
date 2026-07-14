from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.response_window_ensemble import run_response_window_ensemble


def test_response_window_preserves_adjacent_large_integer_labels(tmp_path: Path) -> None:
    first_label = 2**53
    second_label = first_label + 1
    probability_columns = (f"prob_class_{first_label}", f"prob_class_{second_label}")
    rows: list[dict[str, object]] = []

    for subject in ("sub-01", "sub-02"):
        for sample_index, true_label in enumerate((first_label, second_label)):
            probabilities = (0.9, 0.1) if true_label == first_label else (0.1, 0.9)
            predicted_label = first_label if probabilities[0] > probabilities[1] else second_label
            rows.append(
                {
                    "subject": subject,
                    "fold": subject,
                    "decoder": "base",
                    "emission_mode": "calibrated",
                    "time": 0.184,
                    "test_time": 0.184,
                    "sample_index": sample_index,
                    "sequence_id": sample_index,
                    "true_label": true_label,
                    "true_class": f"class-{true_label}",
                    "predicted_label": predicted_label,
                    "predicted_class": f"class-{predicted_label}",
                    "probability_true_class": 0.9,
                    "confidence": 0.9,
                    f"class_{first_label}": f"class-{first_label}",
                    f"class_{second_label}": f"class-{second_label}",
                    probability_columns[0]: probabilities[0],
                    probability_columns[1]: probabilities[1],
                }
            )

    csv_path = tmp_path / "observations.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    ensembled, metrics = run_response_window_ensemble(
        [csv_path],
        mode="uniform",
        response_times=(0.184,),
    )

    assert set(ensembled["predicted_label"].tolist()) == {first_label, second_label}
    assert ensembled["is_correct"].all()
    np.testing.assert_allclose(ensembled["probability_true_class"], 0.9)
    np.testing.assert_allclose(metrics["balanced_accuracy"], 1.0)
