import pandas as pd

from neureptrace.onset_detection import annotate_threshold_crossings, detect_onsets


def _mixed_column_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "subject": ["sub-01"] * 4,
            "sequence_id": [0] * 4,
            "time": [-0.2, -0.1, 0.1, 0.2],
            "true_label": [0] * 4,
            "class_0": ["left"] * 4,
            "class_1": ["right"] * 4,
            "prob_class_0": [0.55, 0.60, 0.90, 0.85],
            "prob_class_1": [0.45, 0.40, 0.10, 0.15],
        }
    )
    frame[7] = ["integer metadata"] * len(frame)
    frame[("metadata", "tag")] = ["tuple metadata"] * len(frame)
    return frame


def test_onset_prediction_inference_ignores_non_string_metadata_columns() -> None:
    frame = _mixed_column_frame()

    thresholded = annotate_threshold_crossings(
        frame,
        threshold_window=(-0.2, -0.1),
        threshold_quantile=1.0,
    )
    events = detect_onsets(
        thresholded,
        threshold_window=(-0.2, -0.1),
        threshold_quantile=1.0,
        detection_start=0.0,
    )

    assert thresholded["predicted_label"].eq(0).all()
    assert thresholded["predicted_class"].eq("left").all()
    assert len(events) == 1
    assert bool(events.loc[0, "detected"])
    assert events.loc[0, "detection_time"] == 0.1
    assert bool(events.loc[0, "is_correct_at_detection"])
