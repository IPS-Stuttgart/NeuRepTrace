import numpy as np

from neureptrace.decoding import HierarchicalThreeClassLogistic


def _training_data():
    features = np.vstack(
        [
            np.column_stack((np.full(8, -2.0), np.linspace(-0.2, 0.2, 8))),
            np.column_stack((np.full(8, 0.0), np.linspace(-0.2, 0.2, 8))),
            np.column_stack((np.full(8, 2.0), np.linspace(-0.2, 0.2, 8))),
        ]
    )
    labels = np.array(
        [["phase", "early"]] * 8
        + [["phase", "middle"]] * 8
        + [["phase", "late"]] * 8,
        dtype=object,
    )
    return features, labels


def _assert_fitted_with_composite_labels(model):
    assert model.classes_.tolist() == [
        ("phase", "early"),
        ("phase", "middle"),
        ("phase", "late"),
    ]
    assert model.primary_class_ == ("phase", "early")
    assert model.second_stage_.classes_.tolist() == [
        ("phase", "middle"),
        ("phase", "late"),
    ]


def test_hierarchical_logistic_preserves_matrix_composite_labels():
    features, labels = _training_data()

    model = HierarchicalThreeClassLogistic(
        primary_class_index=0,
        max_iter=500,
        random_state=0,
    ).fit(features, labels)

    _assert_fitted_with_composite_labels(model)
    probabilities = model.predict_proba(features[:4])
    assert probabilities.shape == (4, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(4))
    assert all(isinstance(label, tuple) for label in model.predict(features[:4]).tolist())


def test_hierarchical_logistic_preserves_object_vector_tuple_labels():
    features, matrix_labels = _training_data()
    labels = np.empty(matrix_labels.shape[0], dtype=object)
    for index, row in enumerate(matrix_labels.tolist()):
        labels[index] = tuple(row)

    model = HierarchicalThreeClassLogistic(
        primary_class_index=0,
        max_iter=500,
        random_state=0,
    ).fit(features, labels)

    _assert_fitted_with_composite_labels(model)
    probabilities = model.predict_proba(features[:4])
    assert probabilities.shape == (4, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(4))
