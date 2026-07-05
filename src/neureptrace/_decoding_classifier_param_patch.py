"""Runtime guardrails for decoder classifier hyperparameters."""

from __future__ import annotations

from typing import Any

import numpy as np

_INTEGER_PATCH_MARKER = "_neureptrace_classifier_integer_params_patch_installed"
_NONNEGATIVE_FLOAT_PARAM_NAMES = {"TorchMLP weight_decay"}
_SCIKIT_MLP_PARAM_ERROR = "scikit-mlp classifier_param must contain hidden_layer_size and max_iter."


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


def _normalize_scikit_mlp_classifier_param(classifier_param: Any) -> tuple[int, int]:
    if isinstance(classifier_param, np.ndarray):
        if classifier_param.ndim == 0:
            raise ValueError(_SCIKIT_MLP_PARAM_ERROR)
        classifier_param = classifier_param.tolist()
    if isinstance(classifier_param, (str, bytes)):
        raise ValueError(_SCIKIT_MLP_PARAM_ERROR)
    try:
        values = tuple(classifier_param)
    except TypeError as exc:
        raise ValueError(_SCIKIT_MLP_PARAM_ERROR) from exc
    if len(values) != 2:
        raise ValueError(_SCIKIT_MLP_PARAM_ERROR)
    hidden_layer_size = _strict_positive_int_classifier_param(values[0], name="scikit-mlp hidden_layer_size")
    max_iter = _strict_positive_int_classifier_param(values[1], name="scikit-mlp max_iter")
    return hidden_layer_size, max_iter


def install() -> None:
    """Install strict classifier-parameter validation in ``neureptrace.decoding``."""

    from neureptrace import decoding as _decoding
    from neureptrace.decoding import classifiers

    if getattr(_decoding, "_positive_float_classifier_param", None) is not _strict_positive_float_classifier_param:
        _decoding._positive_float_classifier_param = _strict_positive_float_classifier_param
    if getattr(classifiers, _INTEGER_PATCH_MARKER, False):
        return

    original_multiclass_svm = classifiers._build_multiclass_svm
    original_multiclass_svm_weighted = classifiers._build_multiclass_svm_weighted
    original_gradient_boosting = classifiers._build_gradient_boosting
    original_knn = classifiers._build_knn
    original_random_forest = classifiers._build_random_forest
    original_scikit_mlp = classifiers._build_scikit_mlp
    original_multinomial_logistic = classifiers._build_multinomial_logistic
    original_multinomial_logistic_weighted = classifiers._build_multinomial_logistic_weighted
    original_shrinkage = classifiers._normalize_lda_shrinkage
    original_xgboost = classifiers._build_xgboost
    original_legacy_gradient_boosting = classifiers.train_gradient_boosting
    original_legacy_lasso_logistic = classifiers.train_lasso_logistic
    original_legacy_binary_svm = classifiers.train_binary_svm

    def build_multiclass_svm(features, labels, classifier_param, random_state):
        classifier_param = _strict_positive_float_classifier_param(classifier_param, default=0.5, name="multiclass-svm classifier_param")
        return original_multiclass_svm(features, labels, classifier_param, random_state)

    def build_multiclass_svm_weighted(features, labels, classifier_param, random_state):
        classifier_param = _strict_positive_float_classifier_param(classifier_param, default=0.5, name="multiclass-svm-weighted classifier_param")
        return original_multiclass_svm_weighted(features, labels, classifier_param, random_state)

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

    def build_scikit_mlp(features, labels, classifier_param, random_state):
        classifier_param = _normalize_scikit_mlp_classifier_param(classifier_param)
        return original_scikit_mlp(features, labels, classifier_param, random_state)

    def build_multinomial_logistic(features, labels, classifier_param, random_state):
        classifier_param = _strict_positive_float_classifier_param(classifier_param, default=1.0, name="multinomial-logistic classifier_param")
        return original_multinomial_logistic(features, labels, classifier_param, random_state)

    def build_multinomial_logistic_weighted(features, labels, classifier_param, random_state):
        classifier_param = _strict_positive_float_classifier_param(classifier_param, default=1.0, name="multinomial-logistic-weighted classifier_param")
        return original_multinomial_logistic_weighted(features, labels, classifier_param, random_state)

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

    def train_lasso_logistic(train_features, train_labels, lambda_, random_state=None):
        lambda_ = _strict_positive_float_classifier_param(lambda_, default=0.005, name="lasso lambda")
        return original_legacy_lasso_logistic(train_features, train_labels, lambda_, random_state)

    def train_for_stimulus_lasso_glm(train_features, train_labels, lambda_, random_state=None):
        return train_lasso_logistic(train_features, train_labels, lambda_, random_state)

    def train_binary_svm(train_features, train_labels, box_constraint, random_state=None):
        box_constraint = _strict_positive_float_classifier_param(box_constraint, default=0.5, name="binary-svm box_constraint")
        return original_legacy_binary_svm(train_features, train_labels, box_constraint, random_state)

    classifiers._build_multiclass_svm = build_multiclass_svm
    classifiers._build_multiclass_svm_weighted = build_multiclass_svm_weighted
    classifiers._build_gradient_boosting = build_gradient_boosting
    classifiers._build_knn = build_knn
    classifiers._build_random_forest = build_random_forest
    classifiers._build_scikit_mlp = build_scikit_mlp
    classifiers._build_multinomial_logistic = build_multinomial_logistic
    classifiers._build_multinomial_logistic_weighted = build_multinomial_logistic_weighted
    classifiers._normalize_lda_shrinkage = normalize_shrinkage
    classifiers._build_xgboost = build_xgboost
    classifiers.train_gradient_boosting = train_gradient_boosting
    classifiers.train_lasso_logistic = train_lasso_logistic
    classifiers.train_for_stimulus_lasso_glm = train_for_stimulus_lasso_glm
    classifiers.train_binary_svm = train_binary_svm
    classifiers.CLASSIFIER_REGISTRY["multiclass-svm"] = classifiers.ClassifierSpec(build_multiclass_svm)
    classifiers.CLASSIFIER_REGISTRY["multiclass-svm-weighted"] = classifiers.ClassifierSpec(build_multiclass_svm_weighted)
    classifiers.CLASSIFIER_REGISTRY["gradient-boosting"] = classifiers.ClassifierSpec(build_gradient_boosting)
    classifiers.CLASSIFIER_REGISTRY["knn"] = classifiers.ClassifierSpec(build_knn)
    classifiers.CLASSIFIER_REGISTRY["random-forest"] = classifiers.ClassifierSpec(build_random_forest)
    classifiers.CLASSIFIER_REGISTRY["scikit-mlp"] = classifiers.ClassifierSpec(build_scikit_mlp)
    classifiers.CLASSIFIER_REGISTRY["multinomial-logistic"] = classifiers.ClassifierSpec(build_multinomial_logistic)
    classifiers.CLASSIFIER_REGISTRY["multinomial-logistic-weighted"] = classifiers.ClassifierSpec(build_multinomial_logistic_weighted)
    classifiers.CLASSIFIER_REGISTRY["xgboost"] = classifiers.ClassifierSpec(build_xgboost)
    setattr(classifiers, _INTEGER_PATCH_MARKER, True)
