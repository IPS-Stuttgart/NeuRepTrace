from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from neureptrace._onset_utils import ensure_prediction_columns, score_values
from neureptrace.temporal_model import _validate_probability_matrix


@pytest.mark.parametrize(
    ("probabilities", "message"),
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
def test_temporal_probability_validation_rejects_silent_float_coercions(
    probabilities: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_probability_matrix(probabilities)


@pytest.mark.parametrize(
    "operation",
    [
        lambda frame: score_values(frame, "confidence"),
        ensure_prediction_columns,
    ],
)
def test_onset_probability_helpers_reject_complex_columns(
    operation: Callable[[pd.DataFrame], object],
) -> None:
    frame = pd.DataFrame(
        {
            "prob_class_0": np.array([0.5 + 0.25j], dtype=np.complex128),
            "prob_class_1": np.array([0.5 - 0.25j], dtype=np.complex128),
        }
    )

    with pytest.raises(ValueError, match="real-valued, not complex"):
        operation(frame)


def test_onset_probability_helpers_keep_existing_predictions_without_revalidating() -> None:
    frame = pd.DataFrame(
        {
            "prob_class_0": np.array([0.5 + 0.25j], dtype=np.complex128),
            "prob_class_1": np.array([0.5 - 0.25j], dtype=np.complex128),
            "predicted_label": [1],
            "predicted_class": ["one"],
        }
    )

    result = ensure_prediction_columns(frame)

    assert result["predicted_label"].tolist() == [1]
    assert result["predicted_class"].tolist() == ["one"]
