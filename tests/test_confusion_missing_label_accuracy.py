from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.metrics import per_class_accuracy


@pytest.mark.parametrize(
    ("true_label", "predicted_label"),
    [
        (float("nan"), np.float64("nan")),
        (np.datetime64("NaT"), np.datetime64("NaT")),
        (np.timedelta64("NaT"), np.timedelta64("NaT")),
        (("session-a", float("nan")), ("session-a", np.float64("nan"))),
    ],
)
def test_per_class_accuracy_matches_equivalent_missing_labels(true_label: object, predicted_label: object) -> None:
    frame = pd.DataFrame(
        {
            "true_label": pd.Series([true_label], dtype=object),
            "predicted_label": pd.Series([predicted_label], dtype=object),
            "participant": ["participant-a"],
        }
    )

    result = per_class_accuracy(frame, participant_column="participant")

    assert result["n_trials"].tolist() == [1]
    assert result["n_correct"].tolist() == [1]
    assert result["accuracy"].tolist() == pytest.approx([1.0])
    assert result["n_participants"].tolist() == [1]


def test_per_class_accuracy_keeps_distinct_missing_label_kinds_separate() -> None:
    frame = pd.DataFrame(
        {
            "true_label": pd.Series([None], dtype=object),
            "predicted_label": pd.Series([float("nan")], dtype=object),
        }
    )

    result = per_class_accuracy(frame)

    assert result["n_trials"].tolist() == [1]
    assert result["n_correct"].tolist() == [0]
    assert result["accuracy"].tolist() == pytest.approx([0.0])
