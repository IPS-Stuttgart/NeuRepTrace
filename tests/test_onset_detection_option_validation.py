from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace import onset_detection


def _minimal_observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["s1", "s1", "s1"],
            "sequence_id": ["trial-1", "trial-1", "trial-1"],
            "time": [-0.1, 0.1, 0.2],
            "confidence": [0.2, 0.9, 0.8],
            "predicted_label": [0, 0, 1],
            "prob_class_0": [0.8, 0.9, 0.2],
            "prob_class_1": [0.2, 0.1, 0.8],
        }
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"threshold_quantile": True}, "threshold_quantile must be a real-valued number"),
        ({"threshold_quantile": np.asarray(0.5)}, "threshold_quantile must be a real-valued number"),
        ({"min_consecutive": True}, "min_consecutive must be a real-valued number"),
        ({"min_duration": False}, "min_duration must be a real-valued number"),
    ],
)
def test_annotate_threshold_crossings_rejects_malformed_numeric_options(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        onset_detection.annotate_threshold_crossings(
            _minimal_observations(),
            threshold_window=(-0.2, -0.05),
            **kwargs,
        )


@pytest.mark.parametrize("value", ["False", "true", 0, 1, np.asarray(False)])
def test_detect_onsets_rejects_non_boolean_stable_prediction_controls(value: object) -> None:
    with pytest.raises(ValueError, match="require_stable_prediction must be a boolean"):
        onset_detection.detect_onsets(
            _minimal_observations(),
            threshold_window=(-0.2, -0.05),
            require_stable_prediction=value,
        )


def test_detect_onsets_accepts_numpy_boolean_stable_prediction() -> None:
    events = onset_detection.detect_onsets(
        _minimal_observations(),
        threshold_window=(-0.2, -0.05),
        require_stable_prediction=np.bool_(False),
    )

    assert len(events) == 1


def test_detect_onsets_from_csvs_rejects_non_boolean_stable_prediction(tmp_path: Path) -> None:
    observations_path = tmp_path / "observations.csv"
    _minimal_observations().to_csv(observations_path, index=False)

    with pytest.raises(ValueError, match="require_stable_prediction must be a boolean"):
        onset_detection.detect_onsets_from_csvs(
            [observations_path],
            threshold_window=(-0.2, -0.05),
            require_stable_prediction="False",
        )
