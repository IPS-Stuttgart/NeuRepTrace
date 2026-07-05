import numpy as np
import pytest

from neureptrace.decoding.classifiers import (
    train_binary_svm,
    train_classifier,
    train_for_stimulus_lasso_glm,
    train_lasso_logistic,
)


FEATURES = np.array(
    [
        [0.0, 0.0],
        [0.0, 0.2],
        [1.0, 1.0],
        [1.0, 1.2],
        [2.0, 2.0],
        [2.0, 2.2],
    ],
    dtype=float,
)
LABELS = np.array([0, 0, 1, 1, 2, 2])
BINARY_FEATURES = np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
BINARY_LABELS = np.array([False, False, True, True])


@pytest.mark.parametrize(
    ("classifier", "classifier_param"),
    [
        ("multiclass-svm", True),
        ("multiclass-svm-weighted", np.asarray(True)),
        ("multinomial-logistic", False),
        ("multinomial-logistic-weighted", np.asarray(False)),
        ("scikit-mlp", (True, 50)),
        ("scikit-mlp", (5, np.asarray(True))),
    ],
)
def test_registry_classifiers_reject_boolean_scalar_params(classifier, classifier_param):
    with pytest.raises(ValueError, match="boolean"):
        train_classifier(FEATURES, LABELS, classifier, classifier_param)


@pytest.mark.parametrize("helper", [train_lasso_logistic, train_for_stimulus_lasso_glm, train_binary_svm])
def test_legacy_binary_helpers_reject_boolean_scalar_params(helper):
    with pytest.raises(ValueError, match="boolean"):
        helper(BINARY_FEATURES, BINARY_LABELS, True)


def test_classifier_scalar_validation_preserves_numeric_numpy_scalars():
    model = train_classifier(FEATURES, LABELS, "multiclass-svm", np.asarray(0.5))

    assert model.predict(FEATURES).shape == LABELS.shape
