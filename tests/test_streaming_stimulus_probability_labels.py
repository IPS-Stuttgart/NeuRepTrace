from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.streaming_stimulus_detection import (
    _probability_columns_from_observation,
    _score_observation,
)


@pytest.mark.parametrize(
    ("probabilities", "target_label"),
    [
        ({"prob_class_2": 0.1, "prob_class_5": 0.9}, 5),
        ({"prob_class_-1": 0.9, "prob_class_2": 0.1}, -1),
    ],
)
@pytest.mark.parametrize(
    "prediction_fields",
    [
        {},
        {"predicted_label": np.nan, "predicted_class": None},
    ],
)
def test_streaming_confidence_inference_preserves_probability_column_labels(
    probabilities: dict[str, float],
    target_label: int,
    prediction_fields: dict[str, object],
) -> None:
    observation = {
        "confidence": 0.9,
        **probabilities,
        **prediction_fields,
    }
    threshold = pd.Series(
        {
            "score_mode": "predicted_class_confidence",
            "score_column": "confidence",
            "stimulus_label": target_label,
            "stimulus_class": f"class-{target_label}",
        }
    )

    assert _score_observation(observation, threshold) == 0.9


def test_streaming_probability_columns_reject_duplicate_signed_label_aliases() -> None:
    observation = {
        "prob_class_-1": 0.5,
        "prob_class_-01": 0.5,
    }

    with pytest.raises(ValueError, match=r"duplicate label\(s\): \[-1\]"):
        _probability_columns_from_observation(observation)
