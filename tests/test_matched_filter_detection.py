import pandas as pd

from neureptrace.matched_filter_detection import _score_values


def test_predicted_class_confidence_fallback_uses_probability_suffix_labels():
    frame = pd.DataFrame(
        {
            "prob_class_1": [0.7, 0.1, 0.6],
            "prob_class_2": [0.3, 0.9, 0.4],
            "confidence": [0.7, 0.9, 0.6],
        }
    )

    class_one_scores = _score_values(
        frame,
        stimulus_label=1,
        stimulus_class="1",
        score_column="prob_class_1",
        score_mode="predicted_class_confidence",
    )
    class_two_scores = _score_values(
        frame,
        stimulus_label=2,
        stimulus_class="2",
        score_column="prob_class_2",
        score_mode="predicted_class_confidence",
    )

    assert class_one_scores.tolist() == [0.7, 0.0, 0.6]
    assert class_two_scores.tolist() == [0.0, 0.9, 0.0]


def test_predicted_class_confidence_fallback_supports_named_probability_suffixes():
    frame = pd.DataFrame(
        {
            "prob_class_face": [0.4, 0.8],
            "prob_class_house": [0.6, 0.2],
            "confidence": [0.6, 0.8],
        }
    )

    face_scores = _score_values(
        frame,
        stimulus_label="face",
        stimulus_class="face",
        score_column="prob_class_face",
        score_mode="predicted_class_confidence",
    )
    house_scores = _score_values(
        frame,
        stimulus_label="house",
        stimulus_class="house",
        score_column="prob_class_house",
        score_mode="predicted_class_confidence",
    )

    assert face_scores.tolist() == [0.0, 0.8]
    assert house_scores.tolist() == [0.6, 0.0]
