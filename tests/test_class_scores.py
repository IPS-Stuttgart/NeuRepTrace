import numpy as np

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
