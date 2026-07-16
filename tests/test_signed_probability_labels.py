from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.observations import probability_columns as observation_probability_columns
from neureptrace.response_window_ensemble import run_response_window_ensemble
from neureptrace.temporal_model import probability_columns as temporal_probability_columns


def test_probability_columns_sort_signed_integer_labels() -> None:
    frame = pd.DataFrame(columns=["prob_class_2", "prob_class_-1"])

    assert temporal_probability_columns(frame) == ["prob_class_-1", "prob_class_2"]
    assert observation_probability_columns(frame) == ("prob_class_-1", "prob_class_2")


@pytest.mark.parametrize("probability_columns", [temporal_probability_columns, observation_probability_columns])
def test_probability_columns_reject_duplicate_signed_integer_aliases(probability_columns) -> None:
    frame = pd.DataFrame(columns=["prob_class_-1", "prob_class_-01"])

    with pytest.raises(ValueError, match=r"duplicate label\(s\): \[-1\]"):
        probability_columns(frame)


def test_response_window_preserves_signed_integer_labels(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for subject in ("sub-01", "sub-02"):
        for sample_index, true_label in enumerate((-1, 2)):
            probability_for_negative = 0.9 if true_label == -1 else 0.1
            probability_for_positive = 1.0 - probability_for_negative
            predicted_label = -1 if probability_for_negative > probability_for_positive else 2
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
                    "prob_class_2": probability_for_positive,
                    "prob_class_-1": probability_for_negative,
                }
            )

    csv_path = tmp_path / "observations.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    ensembled, metrics = run_response_window_ensemble(
        [csv_path],
        mode="uniform",
        response_times=(0.184,),
    )

    assert set(ensembled["predicted_label"].tolist()) == {-1, 2}
    assert set(ensembled["predicted_class"].tolist()) == {"-1", "2"}
    assert ensembled["is_correct"].all()
    np.testing.assert_allclose(ensembled["probability_true_class"], 0.9)
    np.testing.assert_allclose(metrics["balanced_accuracy"], 1.0)
