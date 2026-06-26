import numpy as np

from neureptrace.decoding.class_scores import class_score_matrix, model_classes


class PredictOnlyArrayLabels:
    def predict(self, features):
        labels = [
            np.asarray(("class-a", 0), dtype=object),
            np.asarray(("class-b", 1), dtype=object),
            np.asarray(("class-a", 0), dtype=object),
        ]
        return labels[: len(features)]


class ProbabilityMatrixClasses:
    classes_ = np.asarray(
        [
            ("class-a", 0),
            ("class-b", 1),
        ],
        dtype=object,
    )

    def predict_proba(self, features):
        return np.tile(np.asarray([[0.25, 0.75]], dtype=float), (len(features), 1))


def test_model_classes_preserves_list_composite_fallback_labels() -> None:
    classes = model_classes(object(), fallback_labels=[["class-a", 0], ["class-b", 1], ["class-a", 0]])

    assert classes is not None
    assert classes.tolist() == [("class-a", 0), ("class-b", 1)]


def test_model_classes_preserves_row_matrix_classes() -> None:
    classes = model_classes(ProbabilityMatrixClasses())

    assert classes is not None
    assert classes.tolist() == [("class-a", 0), ("class-b", 1)]


def test_prediction_fallback_matches_array_composite_labels() -> None:
    scores, classes = class_score_matrix(
        PredictOnlyArrayLabels(),
        np.zeros((3, 2)),
        fallback_labels=np.asarray(
            [
                ("class-a", 0),
                ("class-b", 1),
                ("class-a", 0),
            ],
            dtype=object,
        ),
        predict_fallback=True,
    )

    assert classes is not None
    assert classes.tolist() == [("class-a", 0), ("class-b", 1)]
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


def test_probability_scores_preserve_row_matrix_class_order() -> None:
    scores, classes = class_score_matrix(
        ProbabilityMatrixClasses(),
        np.zeros((3, 2)),
    )

    assert classes is not None
    assert classes.tolist() == [("class-a", 0), ("class-b", 1)]
    assert np.array_equal(scores, np.tile(np.asarray([[0.25, 0.75]], dtype=float), (3, 1)))
