import numpy as np

from neureptrace._object_label_utils import values_equal
from neureptrace.decoding.class_scores import class_score_matrix, model_classes


def _object_label_vector(labels: list[tuple[object, str]]) -> np.ndarray:
    vector = np.empty(len(labels), dtype=object)
    for index, label in enumerate(labels):
        vector[index] = label
    return vector


class PredictOnlyTemporalLabels:
    def predict(self, features):
        labels = [(np.datetime64("NaT"), "event"), (None, "event")]
        return _object_label_vector(labels[: len(features)])


def test_model_classes_keeps_composite_nat_distinct_from_none() -> None:
    classes = model_classes(
        object(),
        fallback_labels=[(None, "event"), (np.datetime64("NaT"), "event")],
    )

    assert classes is not None
    assert classes.shape == (2,)
    assert classes[0] == (None, "event")
    assert isinstance(classes[1][0], np.datetime64)
    assert np.isnat(classes[1][0])
    assert not values_equal(classes[0], classes[1])


def test_prediction_fallback_assigns_composite_nat_to_its_class() -> None:
    scores, classes = class_score_matrix(
        PredictOnlyTemporalLabels(),
        np.zeros((2, 1)),
        fallback_labels=[(None, "event"), (np.datetime64("NaT"), "event")],
        predict_fallback=True,
    )

    assert classes is not None
    assert np.array_equal(scores, np.asarray([[0.0, 1.0], [1.0, 0.0]]))
