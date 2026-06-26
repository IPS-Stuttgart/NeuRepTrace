from pathlib import Path

import pandas as pd
import pytest

from neureptrace.temporal_smoothing import smooth_probability_observations


def _row(sequence_id: int, time: float, p0: float, *, true_label: int = 0) -> dict[str, object]:
    p1 = 1.0 - p0
    predicted_label = 0 if p0 >= p1 else 1
    return {
        "time": time,
        "sequence_id": sequence_id,
        "subject": "sub-01",
        "decoder": "logistic",
        "emission_mode": "calibrated",
        "true_label": true_label,
        "class_0": "left",
        "class_1": "right",
        "predicted_label": predicted_label,
        "predicted_class": "left" if predicted_label == 0 else "right",
        "confidence": max(p0, p1),
        "probability_true_class": p0 if true_label == 0 else p1,
        "is_correct": predicted_label == true_label,
        "prob_class_0": p0,
        "prob_class_1": p1,
    }


def test_temporal_smoothing_preserves_singleton_sequences(tmp_path: Path) -> None:
    rows = []
    for sequence_id in range(6):
        rows.append(_row(sequence_id, 0.10, 0.90))
        rows.append(_row(sequence_id, 0.20, 0.86))
    rows.append(_row(99, 0.15, 0.25, true_label=1))

    csv_path = tmp_path / "observations.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    smoothed, metrics = smooth_probability_observations(
        [csv_path],
        fit_window=(0.10, 0.20),
        stay_grid_size=20,
    )

    singleton = smoothed.loc[smoothed["sequence_id"].eq(99)].iloc[0]
    assert len(smoothed) == len(rows)
    assert metrics["n_test"].sum() == len(rows)
    assert singleton["prob_class_0"] == pytest.approx(0.25)
    assert singleton["prob_class_1"] == pytest.approx(0.75)
    assert singleton["predicted_label"] == 1
    assert singleton["is_correct"]
    assert singleton["emission_mode"] == "calibrated_temporal_posterior"
    assert singleton["temporal_smoothing_method"] == "sticky_forward_backward"
