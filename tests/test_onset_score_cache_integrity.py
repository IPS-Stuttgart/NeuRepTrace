from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.onset_detection import annotate_threshold_crossings, detect_onsets


def _single_sequence_frame(scores: list[tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for time, confidence in scores:
        rows.append(
            {
                "subject": "sub-01",
                "decoder": "logistic",
                "emission_mode": "calibrated",
                "time": time,
                "sequence_id": 0,
                "sample_index": 0,
                "confidence": confidence,
                "prob_class_0": confidence,
                "prob_class_1": 1.0 - confidence,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("bad_score", ["not-a-number", np.inf, True])
def test_annotate_threshold_crossings_rejects_invalid_explicit_scores(bad_score: object) -> None:
    observations = _single_sequence_frame(
        [(-0.30, 0.20), (-0.20, 0.40), (0.10, 0.80)]
    )
    observations["custom_score"] = [0.20, bad_score, 0.80]

    with pytest.raises(ValueError, match="custom_score values must be finite numeric scores"):
        annotate_threshold_crossings(
            observations,
            threshold_window=(-0.30, -0.20),
            threshold_quantile=1.0,
            score_column="custom_score",
        )


@pytest.mark.parametrize(
    ("metadata_column", "missing_value"),
    [
        ("score_column", np.nan),
        ("threshold_method", np.nan),
        ("require_stable_prediction", np.nan),
    ],
)
def test_detect_onsets_recomputes_cache_with_incomplete_metadata(
    metadata_column: str,
    missing_value: object,
) -> None:
    observations = _single_sequence_frame(
        [(-0.30, 0.20), (-0.20, 0.40), (0.10, 0.50)]
    )
    cached = annotate_threshold_crossings(
        observations,
        threshold_window=(-0.30, -0.20),
        threshold_quantile=1.0,
    )
    cached["score_threshold"] = 0.99
    cached[metadata_column] = cached[metadata_column].astype(object)
    cached.loc[cached.index[0], metadata_column] = missing_value

    events = detect_onsets(
        cached,
        threshold_window=(-0.30, -0.20),
        threshold_quantile=1.0,
        detection_start=0.0,
    )

    event = events.iloc[0]
    assert bool(event["detected"])
    assert event["detection_time"] == pytest.approx(0.10)
    assert event["score_threshold"] == pytest.approx(0.40)
