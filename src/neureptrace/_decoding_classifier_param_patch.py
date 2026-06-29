"""Runtime guardrails for decoder classifier hyperparameters."""

from __future__ import annotations

from typing import Any

import numpy as np

_INTEGER_PATCH_MARKER = "_neureptrace_classifier_integer_params_patch_installed"
_NONNEGATIVE_FLOAT_PARAM_NAMES = {"TorchMLP weight_decay"}


def _scalar_classifier_param(
    classifier_param: Any,
    *,
    name: str,
    scalar_kind: str,
    boolean_message: str,
) -> Any:
    """Return a scalar classifier parameter without ndarray/boolean coercion."""

    if isinstance(classifier_param, (bool, np.bool_)):
        raise ValueError(boolean_message)
    if isinstance(classifier_param, np.ndarray):
        if np.issubdtype(classifier_param.dtype, np.bool_):
            raise ValueError(boolean_message)
        if classifier_param.ndim != 0:
            raise ValueError(f"{name} must be a scalar {scalar_kind} value.")
        classifier_param = classifier_param.item()
        if isinstance(classifier_param, (bool, np.bool_)):
            raise ValueError(boolean_message)
    return classifier_param


def _strict_positive_float_classifier_param(
    classifier_param: Any,
    *,
    default: float,
    name: str,
) -> float:
    """Normalize float classifier parameters without bool coercion.

    Most decoder scalar parameters are regularization strengths such as ``C`` and
    therefore must be strictly positive.  ``torch_mlp`` is the exception: its
    scalar ``classifier_param`` is forwarded as weight decay, where zero is a
    valid value to disable the penalty.
    """

    if classifier_param is None:
        value = float(default)
    else:
        classifier_param = _scalar_classifier_param(
            classifier_param,
            name=name,
            scalar_kind="numeric",
            boolean_message=f"{name} must be numeric, not boolean.",
        )
        try:
            value = float(classifier_param)
        except (TypeError, ValueError) as exc:
            adjective = "non-negative" if name in _NONNEGATIVE_FLOAT_PARAM_NAMES else "positive"
            raise ValueError(f"{name} must be a {adjective} finite value.") from exc

    if name in _NONNEGATIVE_FLOAT_PARAM_NAMES:
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a non-negative finite value.")
    elif not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
    return value


def _strict_positive_int_classifier_param(classifier_param: Any, *, name: str) -> int:
    classifier_param = _scalar_classifier_param(
        classifier_param,
        name=name,
        scalar_kind="positive integer",
        boolean_message=f"{name} must be a positive integer, not boolean.",
    )
    try:
        value = float(classifier_param)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(value) or not value.is_integer() or value <= 0.0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def install() -> None:
    """Install strict classifier-parameter validation in ``neureptrace.decoding``."""

    from neureptrace import decoding as _decoding
    from neureptrace.decoding import classifiers

    if getattr(_decoding, "_positive_float_classifier_param", None) is not _strict_positive_float_classifier_param:
        _decoding._positive_float_classifier_param = _strict_positive_float_classifier_param
    if getattr(classifiers, _INTEGER_PATCH_MARKER, False):
        return

    original_gradient_boosting = classifiers._build_gradient_boosting
    original_knn = classifiers._build_knn
    original_random_forest = classifiers._build_random_forest
    original_shrinkage = classifiers._normalize_lda_shrinkage
    original_xgboost = classifiers._build_xgboost
    original_legacy_gradient_boosting = classifiers.train_gradient_boosting

    def build_gradient_boosting(features, labels, classifier_param, random_state):
        classifier_param = _strict_positive_int_classifier_param(classifier_param, name="gradient-boosting classifier_param")
        return original_gradient_boosting(features, labels, classifier_param, random_state)

    def build_knn(features, labels, classifier_param, random_state):
        requested_neighbors = _strict_positive_int_classifier_param(classifier_param, name="knn classifier_param")
        features_array = np.asarray(features)
        if features_array.ndim != 2:
            raise ValueError("knn classifier features must be a two-dimensional matrix.")
        n_training_rows = int(features_array.shape[0])
        if n_training_rows < 1:
            raise ValueError("knn classifier requires at least one training sample.")
        classifier_param = min(requested_neighbors, n_training_rows)
        return original_knn(features, labels, classifier_param, random_state)

    def build_random_forest(features, labels, classifier_param, random_state):
        classifier_param = _strict_positive_int_classifier_param(classifier_param, name="random-forest classifier_param")
        return original_random_forest(features, labels, classifier_param, random_state)

    def normalize_shrinkage(classifier_param):
        classifier_param = _scalar_classifier_param(
            classifier_param,
            name="shrinkage-lda classifier_param",
            scalar_kind="numeric",
            boolean_message="shrinkage-lda classifier_param must be numeric, not boolean.",
        )
        return original_shrinkage(classifier_param)

    def build_xgboost(features, labels, classifier_param, random_state):
        classifier_param = _strict_positive_int_classifier_param(classifier_param, name="xgboost classifier_param")
        return original_xgboost(features, labels, classifier_param, random_state)

    def train_gradient_boosting(train_features, train_labels, classifier_param, random_state=None):
        classifier_param = _strict_positive_int_classifier_param(classifier_param, name="gradient_boosting classifier_param")
        return original_legacy_gradient_boosting(train_features, train_labels, classifier_param, random_state)

    classifiers._build_gradient_boosting = build_gradient_boosting
    classifiers._build_knn = build_knn
    classifiers._build_random_forest = build_random_forest
    classifiers._normalize_lda_shrinkage = normalize_shrinkage
    classifiers._build_xgboost = build_xgboost
    classifiers.train_gradient_boosting = train_gradient_boosting
    classifiers.CLASSIFIER_REGISTRY["gradient-boosting"] = classifiers.ClassifierSpec(build_gradient_boosting)
    classifiers.CLASSIFIER_REGISTRY["knn"] = classifiers.ClassifierSpec(build_knn)
    classifiers.CLASSIFIER_REGISTRY["random-forest"] = classifiers.ClassifierSpec(build_random_forest)
    classifiers.CLASSIFIER_REGISTRY["xgboost"] = classifiers.ClassifierSpec(build_xgboost)
    setattr(classifiers, _INTEGER_PATCH_MARKER, True)
