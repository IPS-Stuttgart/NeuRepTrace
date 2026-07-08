import numpy as np
import pytest

from neureptrace.decoding.class_scores import class_score_matrix, model_classes


def _tuple_label_vector(labels: list[tuple[str, int]]) -> np.ndarray:
    vector = np.empty(len(labels), dtype=object)
    for index, label in enumerate(labels):
        vector[index] = label
    return vector


class PredictOnlyCompositeLabels:
    def predict(self, features):
        labels = [("visual", 0), ("auditory", 1), ("visual", 0)]
        return _tuple_label_vector(labels[: len(features)])


class BinaryDecisionScores:
    classes_ = np.asarray(["left", "right"])

    def decision_function(self, features):
        return np.asarray(features)[:, 0]


def _nested_feature_generators(rows):
    return ((value for value in row) for row in rows)


def test_model_classes_preserves_composite_fallback_labels() -> None:
    classes = model_classes(
        object(),
        fallback_labels=[("visual", 0), ("auditory", 1), ("visual", 0)],
    )

    assert classes is not None
    assert classes.tolist() == [("visual", 0), ("auditory", 1)]


def test_prediction_fallback_preserves_composite_class_order() -> None:
    scores, classes = class_score_matrix(
        PredictOnlyCompositeLabels(),
        np.zeros((3, 2)),
        fallback_labels=[("visual", 0), ("auditory", 1), ("visual", 0)],
        predict_fallback=True,
    )

    assert classes is not None
    assert classes.tolist() == [("visual", 0), ("auditory", 1)]
    assert np.array_equal(
        scores,
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ),
    )


def test_class_score_matrix_accepts_nested_one_pass_feature_rows() -> None:
    scores, classes = class_score_matrix(
        BinaryDecisionScores(),
        _nested_feature_generators([[-1.0], [2.0]]),
    )

    assert classes is not None
    assert classes.tolist() == ["left", "right"]
    assert np.array_equal(scores, np.asarray([[1.0, -1.0], [-2.0, 2.0]]))


@pytest.mark.parametrize(
    "features",
    [
        np.asarray([[True], [False]]),
        [[False], [True]],
        _nested_feature_generators([[False], [True]]),
    ],
)
def test_class_score_matrix_rejects_boolean_feature_values(features) -> None:
    with pytest.raises(ValueError, match="non-boolean"):
        class_score_matrix(BinaryDecisionScores(), features)
