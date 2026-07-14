from __future__ import annotations

import pandas as pd

from neureptrace.onset_detection import annotate_threshold_crossings, detect_onsets


def _large_label_observations() -> pd.DataFrame:
    first_label = 2**53
    second_label = first_label + 1
    rows: list[dict[str, object]] = []

    for sequence_id, predicted_label in ((0, first_label), (1, second_label)):
        for time, confidence in ((-0.2, 0.5), (0.1, 0.9)):
            rows.append(
                {
                    "subject": "sub-01",
                    "decoder": "logistic",
                    "emission_mode": "calibrated",
                    "sequence_id": sequence_id,
                    "time": time,
                    "confidence": confidence,
                    "true_label": second_label,
                    "true_class": f"class-{second_label}",
                    "predicted_label": predicted_label,
                    "predicted_class": f"class-{predicted_label}",
                }
            )
    return pd.DataFrame(rows)


def test_onset_correctness_preserves_adjacent_large_integer_labels() -> None:
    observations = _large_label_observations()

    thresholded = annotate_threshold_crossings(
        observations,
        threshold_window=(-0.2, -0.2),
        threshold_quantile=0.5,
    )
    positive_rows = thresholded.loc[thresholded["time"] > 0.0].set_index("sequence_id")

    assert not bool(positive_rows.loc[0, "is_correct"])
    assert bool(positive_rows.loc[1, "is_correct"])

    events = detect_onsets(
        observations,
        threshold_window=(-0.2, -0.2),
        threshold_quantile=0.5,
        detection_start=0.0,
    ).set_index("sequence_id")

    assert events["detected"].all()
    assert not bool(events.loc[0, "is_correct_at_detection"])
    assert bool(events.loc[1, "is_correct_at_detection"])
