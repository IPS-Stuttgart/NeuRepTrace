import numpy as np
import pytest

from neureptrace.decoding.vrex import LinearVRExClassifier


def test_vrex_accepts_one_pass_feature_rows_for_fit_and_prediction():
    source_rows = ([10.0, 2.0], [11.0, 2.5], [12.0, 3.0], [13.0, 3.5])
    source_features = (row for row in source_rows)
    source_labels = (label for label in ["left", "right", "left", "right"])
    source_domains = (domain for domain in ["s1", "s1", "s2", "s2"])

    model = LinearVRExClassifier(max_iter=3, tol=1e-4)
    model.fit(source_features, source_labels, source_domains=source_domains)
    probabilities = model.predict_proba((row for row in ([10.5, 2.2], [12.5, 3.2])))

    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert model.n_features_in_ == 2


def test_vrex_accepts_nested_one_pass_feature_rows_for_fit_and_prediction():
    source_rows = ([10.0, 2.0], [11.0, 2.5], [12.0, 3.0], [13.0, 3.5])
    source_features = (iter(row) for row in source_rows)
    source_labels = (label for label in ["left", "right", "left", "right"])
    source_domains = (domain for domain in ["s1", "s1", "s2", "s2"])

    model = LinearVRExClassifier(max_iter=3, tol=1e-4)
    model.fit(source_features, source_labels, source_domains=source_domains)
    probabilities = model.predict_proba((iter(row) for row in ([10.5, 2.2], [12.5, 3.2])))

    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert model.n_features_in_ == 2


def test_vrex_reports_matrix_error_for_bad_one_pass_feature_rows():
    source_features = (row for row in ([10.0, 2.0], [11.0]))
    source_labels = ["left", "right"]
    source_domains = ["s1", "s2"]

    model = LinearVRExClassifier(max_iter=3, tol=1e-4)

    with pytest.raises(ValueError, match="source_features must be a non-empty two-dimensional feature matrix"):
        model.fit(source_features, source_labels, source_domains=source_domains)


@pytest.mark.parametrize(
    "source_features",
    [
        [[False, True], [True, False], [False, True], [True, False]],
        np.asarray([[False, True], [True, False], [False, True], [True, False]], dtype=bool),
        np.asarray([[False, 1.0], [True, 0.0], [False, 1.0], [True, 0.0]], dtype=object),
        (iter(row) for row in ([False, 1.0], [True, 0.0], [False, 1.0], [True, 0.0])),
    ],
)
def test_vrex_rejects_boolean_fit_features(source_features):
    model = LinearVRExClassifier(max_iter=3, tol=1e-4)

    with pytest.raises(ValueError, match="source_features.*boolean flags"):
        model.fit(
            source_features,
            ["left", "right", "left", "right"],
            source_domains=["s1", "s1", "s2", "s2"],
        )


def test_vrex_rejects_boolean_prediction_features():
    source_features = np.asarray([[10.0, 2.0], [11.0, 2.5], [12.0, 3.0], [13.0, 3.5]])
    model = LinearVRExClassifier(max_iter=3, tol=1e-4)
    model.fit(source_features, ["left", "right", "left", "right"], source_domains=["s1", "s1", "s2", "s2"])

    with pytest.raises(ValueError, match="features.*boolean flags"):
        model.predict_proba(np.asarray([[True, False]], dtype=bool))
