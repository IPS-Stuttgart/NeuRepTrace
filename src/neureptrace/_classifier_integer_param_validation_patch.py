"""Reject boolean values for integer-valued classifier hyperparameters."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_classifier_integer_params_patch_installed"


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer, not boolean.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(numeric)


def install() -> None:
    """Install classifier integer-parameter validation guards."""

    from neureptrace.decoding import classifiers

    if getattr(classifiers, _PATCH_MARKER, False):
        return

    original_build_gradient_boosting = classifiers._build_gradient_boosting
    original_build_knn = classifiers._build_knn
    original_build_random_forest = classifiers._build_random_forest
    original_normalize_lda_shrinkage = classifiers._normalize_lda_shrinkage
    original_train_gradient_boosting = classifiers.train_gradient_boosting

    @wraps(original_build_gradient_boosting)
    def build_gradient_boosting(features, labels, classifier_param, random_state):
        return original_build_gradient_boosting(
            features,
            labels,
            _positive_int(classifier_param, name="gradient-boosting classifier_param"),
            random_state,
        )

    @wraps(original_build_knn)
    def build_knn(features, labels, classifier_param, random_state):
        return original_build_knn(
            features,
            labels,
            _positive_int(classifier_param, name="knn classifier_param"),
            random_state,
        )

    @wraps(original_build_random_forest)
    def build_random_forest(features, labels, classifier_param, random_state):
        return original_build_random_forest(
            features,
            labels,
            _positive_int(classifier_param, name="random-forest classifier_param"),
            random_state,
        )

    @wraps(original_normalize_lda_shrinkage)
    def normalize_lda_shrinkage(classifier_param):
        if isinstance(classifier_param, (bool, np.bool_)):
            raise ValueError("shrinkage-lda classifier_param must be numeric, not boolean.")
        return original_normalize_lda_shrinkage(classifier_param)

    @wraps(original_train_gradient_boosting)
    def train_gradient_boosting(train_features, train_labels, classifier_param, random_state=None):
        return original_train_gradient_boosting(
            train_features,
            train_labels,
            _positive_int(classifier_param, name="gradient_boosting classifier_param"),
            random_state,
        )

    classifiers._build_gradient_boosting = build_gradient_boosting
    classifiers._build_knn = build_knn
    classifiers._build_random_forest = build_random_forest
    classifiers._normalize_lda_shrinkage = normalize_lda_shrinkage
    classifiers.train_gradient_boosting = train_gradient_boosting

    classifiers.CLASSIFIER_REGISTRY["gradient-boosting"] = classifiers.ClassifierSpec(build_gradient_boosting)
    classifiers.CLASSIFIER_REGISTRY["knn"] = classifiers.ClassifierSpec(build_knn)
    classifiers.CLASSIFIER_REGISTRY["random-forest"] = classifiers.ClassifierSpec(build_random_forest)
    setattr(classifiers, _PATCH_MARKER, True)
