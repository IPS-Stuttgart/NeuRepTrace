from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.classifiers import train_classifier, train_gradient_boosting, train_multiclass_classifier


@pytest.fixture
def multiclass_data():
    features = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.2],
            [1.0, 1.0],
            [1.0, 1.2],
            [2.0, 2.0],
            [2.0, 2.2],
        ]
    )
    labels = np.array([0, 0, 1, 1, 2, 2])
    return features, labels


@pytest.mark.parametrize("classifier", ["gradient-boosting", "knn", "random-forest"])
@pytest.mark.parametrize("classifier_param", [True, False, np.bool_(True), np.bool_(False)])
def test_train_classifier_rejects_boolean_integer_params(multiclass_data, classifier, classifier_param):
    features, labels = multiclass_data
    with pytest.raises(ValueError, match="integer, not boolean"):
        train_classifier(features, labels, classifier, classifier_param)


@pytest.mark.parametrize("classifier_param", [True, False, np.bool_(True), np.bool_(False)])
def test_shrinkage_lda_rejects_boolean_classifier_param(multiclass_data, classifier_param):
    features, labels = multiclass_data
    with pytest.raises(ValueError, match="numeric, not boolean"):
        train_multiclass_classifier(features, labels, "shrinkage-lda", classifier_param)


@pytest.mark.parametrize("classifier_param", [True, False, np.bool_(True), np.bool_(False)])
def test_legacy_gradient_boosting_rejects_boolean_classifier_param(classifier_param):
    features = np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    labels = np.array([False, False, True, True])
    with pytest.raises(ValueError, match="integer, not boolean"):
        train_gradient_boosting(features, labels, classifier_param)
