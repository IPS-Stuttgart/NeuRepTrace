from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from neureptrace import onset_detection
from neureptrace._onset_utils import ensure_prediction_columns, score_values


@pytest.mark.parametrize(
    "operation",
    [
        lambda frame: score_values(frame, "confidence"),
        ensure_prediction_columns,
        lambda frame: onset_detection._score_values(frame, "confidence"),
        onset_detection._ensure_prediction_columns,
    ],
)
@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            np.array([[0.5 + 0.25j, 0.5 - 0.25j]], dtype=np.complex128),
            "real-valued, not complex",
        ),
        (
            np.array([[True, False]], dtype=bool),
            "numeric, not boolean",
        ),
    ],
)
def test_onset_probability_helpers_reject_silent_float_coercions(
    operation: Callable[[pd.DataFrame], object],
    values: np.ndarray,
    message: str,
) -> None:
    frame = pd.DataFrame(
        {
            "prob_class_0": values[:, 0],
            "prob_class_1": values[:, 1],
        }
    )

    with pytest.raises(ValueError, match=message):
        operation(frame)


@pytest.mark.parametrize(
    "operation",
    [
        ensure_prediction_columns,
        onset_detection._ensure_prediction_columns,
    ],
)
def test_onset_probability_helpers_keep_complete_existing_predictions(
    operation: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    frame = pd.DataFrame(
        {
            "prob_class_0": np.array([0.5 + 0.25j], dtype=np.complex128),
            "prob_class_1": np.array([0.5 - 0.25j], dtype=np.complex128),
            "predicted_label": [1],
            "predicted_class": ["one"],
        }
    )

    result = operation(frame)

    assert result["predicted_label"].tolist() == [1]
    assert result["predicted_class"].tolist() == ["one"]
