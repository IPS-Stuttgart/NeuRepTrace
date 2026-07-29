# ruff: noqa
"""Composite-label support and robust validation for windowed decoding helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from neureptrace._object_label_utils import label_accuracy, label_counts, values_equal

_INSTALLED = False
_ORIGINAL_LABEL_VECTOR = None
_ORIGINAL_CLASS_SCORE_MATRIX = None
_ORIGINAL_AS_CLASS_SCORE_MATRIX = None
_ORIGINAL_FEATURE_MATRIX = None


def _object_vector(values) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _is_composite_label(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim > 0
    return isinstance(value, (list, tuple))


def _as_atomic_label(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=object)
        if array.ndim == 0:
            return _as_atomic_label(array.item())
        return tuple(_as_atomic_label(item) for item in array.reshape(-1).tolist())
    if isinstance(value, list):
        return tuple(_as_atomic_label(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_as_atomic_label(item) for item in value)
    return value


def _sequence_atomic_vector(values: Sequence | np.ndarray) -> np.ndarray | None:
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=object)
        if array.ndim <= 0:
            return None
        if array.ndim == 1:
            items = array.tolist()
            if not items or not any(_is_composite_label(value) for value in items):
                return None
            return _object_vector(_as_atomic_label(value) for value in items)
        rows = array.reshape(array.shape[0], -1)
        if rows.shape[1] == 1:
            return None
        return _object_vector(tuple(_as_atomic_label(item) for item in row.tolist()) for row in rows)
    items = list(values)
    if not items or not any(_is_composite_label(value) for value in items):
        return None
    return _object_vector(_as_atomic_label(value) for value in items)


def _materialize_label_input(values: Sequence | np.ndarray) -> Sequence | np.ndarray:
    """Keep reusable labels intact and materialize one-pass iterables once."""

    if isinstance(values, (np.ndarray, str, bytes, Sequence)):
        return values
    return list(values)


def _label_vector(labels: Sequence | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = _sequence_atomic_vector(labels)
    if vector is None:
        return _ORIGINAL_LABEL_VECTOR(labels, expected_length=expected_length, name=name)
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} length must match feature rows: {vector.shape[0]} != {expected_length}.")
    return vector


def _prediction_vector(predictions: Sequence | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = _sequence_atomic_vector(predictions)
    if vector is None:
        try:
            array = np.asarray(predictions)
        except (TypeError, ValueError):
            vector = _object_vector(_as_atomic_label(value) for value in predictions)
        else:
            if array.ndim == 0:
                vector = np.asarray([_as_atomic_label(array.item())], dtype=object)
            elif array.ndim == 1:
                vector = array.copy()
            elif 1 in array.shape:
                vector = array.reshape(-1).copy()
            elif array.dtype == object:
                rows = np.asarray(array, dtype=object).reshape(array.shape[0], -1)
                vector = _object_vector(tuple(_as_atomic_label(item) for item in row.tolist()) for row in rows)
            else:
                raise ValueError(f"{name} must be one-dimensional.")
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} length must match feature rows: {vector.shape[0]} != {expected_length}.")
    return vector


def _label_equal_mask(labels: Sequence | np.ndarray, label: object) -> np.ndarray:
    return np.asarray([values_equal(value, label) for value in labels], dtype=bool)


def _balanced_accuracy(predictions: Sequence | np.ndarray, labels: Sequence | np.ndarray) -> float:
    labels = _label_vector(labels, expected_length=len(labels), name="labels")
    predictions = _prediction_vector(predictions, expected_length=len(labels), name="predictions")
    if labels.size == 0:
        return np.nan
    label_values = label_counts(labels)[0]
    recalls = []
    for label in label_values:
        mask = _label_equal_mask(labels, label)
        if np.any(mask):
            recalls.append(label_accuracy(labels[mask], predictions[mask]))
    return float(np.mean(recalls)) if recalls else np.nan


def _materialize_feature_input(values: object) -> object:
    """Return feature inputs that NumPy can inspect without consuming generators."""

    if isinstance(values, np.ndarray):
        if values.dtype != object:
            return values
        return _materialize_feature_input(values.tolist())
    if isinstance(values, (str, bytes)):
        return values
    if hasattr(values, "__array__"):
        return values
    if not isinstance(values, Iterable):
        return values
    return [_materialize_feature_input(value) for value in values]


def _features_contain_boolean(values: object) -> bool:
    if isinstance(values, (bool, np.bool_)):
        return True
    if isinstance(values, np.ndarray):
        if np.issubdtype(values.dtype, np.bool_):
            return bool(values.size)
        if values.dtype == object:
            return any(_features_contain_boolean(value) for value in values.ravel(order="C"))
        return False
    if hasattr(values, "__array__"):
        try:
            return _features_contain_boolean(np.asarray(values, dtype=object))
        except (TypeError, ValueError):
            return False
    if isinstance(values, (str, bytes)):
        return False
    if not isinstance(values, Iterable):
        return False
    return any(_features_contain_boolean(value) for value in values)


def _features_contain_complex(values: object) -> bool:
    if isinstance(values, (complex, np.complexfloating)):
        return True
    if isinstance(values, np.ndarray):
        if np.issubdtype(values.dtype, np.complexfloating):
            return bool(values.size)
        if values.dtype == object:
            return any(_features_contain_complex(value) for value in values.ravel(order="C"))
        return False
    if hasattr(values, "__array__"):
        try:
            return _features_contain_complex(np.asarray(values))
        except (TypeError, ValueError):
            return False
    if isinstance(values, (str, bytes)):
        return False
    if not isinstance(values, Iterable):
        return False
    return any(_features_contain_complex(value) for value in values)


def _feature_matrix(features: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    materialized = _materialize_feature_input(features)
    if _features_contain_complex(materialized):
        raise ValueError(f"{name} must contain real-valued features, not complex values.")
    return _ORIGINAL_FEATURE_MATRIX(materialized, name=name)


def _class_score_feature_matrix(features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    materialized = _materialize_feature_input(features)
    if _features_contain_boolean(materialized):
        raise ValueError("features must contain numeric, non-boolean values.")
    if _features_contain_complex(materialized):
        raise ValueError("features must contain real-valued features, not complex values.")
    try:
        matrix = np.asarray(materialized, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("features must be a two-dimensional numeric feature matrix.") from exc
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional feature matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must contain only finite values.")
    return matrix


def install() -> None:
    """Install the composite-label-safe windowed decoding patch."""

    global _INSTALLED, _ORIGINAL_LABEL_VECTOR, _ORIGINAL_CLASS_SCORE_MATRIX, _ORIGINAL_AS_CLASS_SCORE_MATRIX, _ORIGINAL_FEATURE_MATRIX
    if _INSTALLED:
        return

    from neureptrace.decoding import class_scores, transfer, windowed

    _ORIGINAL_LABEL_VECTOR = windowed._label_vector
    _ORIGINAL_FEATURE_MATRIX = windowed._feature_matrix
    windowed._label_vector = _label_vector
    windowed._feature_matrix = _feature_matrix
    windowed._balanced_accuracy = _balanced_accuracy

    original_predict_window_model = windowed.predict_window_model
    original_score_windowed_decoding = windowed.score_windowed_decoding
    original_permutation_score_curves = windowed.permutation_score_curves
    original_permutation_p_from_accuracy = windowed.permutation_p_from_accuracy
    original_cross_validate_feature_decoding = transfer.cross_validate_feature_decoding
    _ORIGINAL_CLASS_SCORE_MATRIX = class_scores.class_score_matrix
    _ORIGINAL_AS_CLASS_SCORE_MATRIX = class_scores.as_class_score_matrix

    def predict_window_model(model_bundle: Any, features: Sequence[Sequence[float]] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transformed_features = windowed.transform_window_features(model_bundle, features)
        predictions = _prediction_vector(
            model_bundle.model.predict(transformed_features),
            expected_length=transformed_features.shape[0],
            name="predictions",
        )
        scores = windowed.prediction_scores(model_bundle.model, transformed_features)
        return predictions, scores

    def score_windowed_decoding(
        train_features: Sequence[Sequence[float]] | np.ndarray,
        train_labels: Sequence | np.ndarray,
        validation_features: Sequence[Sequence[float]] | np.ndarray,
        validation_labels: Sequence | np.ndarray,
        *,
        fit_model: windowed.FitModel,
        components_pca: int | float | str | None = float("inf"),
        train_window: tuple[float, float] | None = None,
        n_permutations: int = 0,
        permutation_rng: np.random.Generator | None = None,
    ) -> windowed.WindowedDecodingResult:
        materialized_validation_labels = _materialize_label_input(validation_labels)
        result = original_score_windowed_decoding(
            train_features,
            train_labels,
            validation_features,
            materialized_validation_labels,
            fit_model=fit_model,
            components_pca=components_pca,
            train_window=train_window,
            n_permutations=n_permutations,
            permutation_rng=permutation_rng,
        )
        normalized_labels = _label_vector(
            materialized_validation_labels,
            expected_length=result.predictions.shape[0],
            name="validation_labels",
        )
        return replace(result, accuracy=label_accuracy(normalized_labels, result.predictions))

    def permutation_score_curves(
        train_features: Sequence[Sequence[float]] | np.ndarray,
        *,
        validation_features: Sequence[Sequence[float]] | np.ndarray,
        validation_labels: Sequence | np.ndarray,
        train_labels: Sequence | np.ndarray,
        fit_model: windowed.FitModel,
        n_permutations: int,
        permutation_rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_permutations = windowed._validate_permutation_count(n_permutations)
        train_features = windowed._feature_matrix(train_features, name="train_features")
        validation_features = windowed._feature_matrix(validation_features, name="validation_features")
        train_labels = _label_vector(train_labels, expected_length=train_features.shape[0], name="train_labels")
        validation_labels = _label_vector(
            validation_labels,
            expected_length=validation_features.shape[0],
            name="validation_labels",
        )
        if permutation_rng is None:
            permutation_rng = np.random.default_rng()

        permuted_accuracy = []
        permuted_balanced_accuracy = []
        for _ in range(n_permutations):
            permuted_train_labels = np.array(train_labels, copy=True)
            permutation_rng.shuffle(permuted_train_labels)
            model = fit_model(train_features, permuted_train_labels)
            predictions = _prediction_vector(
                model.predict(validation_features),
                expected_length=validation_features.shape[0],
                name="predictions",
            )
            permuted_accuracy.append(label_accuracy(validation_labels, predictions))
            permuted_balanced_accuracy.append(_balanced_accuracy(predictions, validation_labels))
        return np.asarray(permuted_accuracy, dtype=float), np.asarray(permuted_balanced_accuracy, dtype=float)

    def permutation_p_from_accuracy(accuracy: float, permutation_accuracy: Sequence[float] | np.ndarray) -> float:
        permutation_accuracy = np.asarray(permutation_accuracy, dtype=float)
        finite_permutation_accuracy = permutation_accuracy[np.isfinite(permutation_accuracy)]
        return original_permutation_p_from_accuracy(accuracy, finite_permutation_accuracy)

    def cross_validate_feature_decoding(
        stimulus_features: Sequence[Sequence[float]] | np.ndarray,
        labels: Sequence | np.ndarray,
        *,
        null_features: Sequence[Sequence[float]] | np.ndarray | None = None,
        n_folds: int = 10,
        classifier: str = "multiclass-svm",
        classifier_param: Any = 0.5,
        components_pca: int | float = float("inf"),
        random_state: int | None = None,
        fit_model: Any = None,
        null_label: int | float = 0,
        generative_augmentation: Any = None,
    ) -> Any:
        materialized_labels = _materialize_label_input(labels)
        result = original_cross_validate_feature_decoding(
            stimulus_features,
            materialized_labels,
            null_features=null_features,
            n_folds=n_folds,
            classifier=classifier,
            classifier_param=classifier_param,
            components_pca=components_pca,
            random_state=random_state,
            fit_model=fit_model,
            null_label=null_label,
            generative_augmentation=generative_augmentation,
        )
        normalized_labels = transfer._label_vector(
            materialized_labels,
            expected_length=result.predictions.shape[0],
            name="labels",
        )
        return replace(result, accuracy=label_accuracy(normalized_labels, result.predictions))

    def as_class_score_matrix(
        raw_scores: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
        classes: Sequence | np.ndarray,
        *,
        n_samples: int,
    ) -> np.ndarray | None:
        materialized = _materialize_feature_input(raw_scores)
        if _features_contain_complex(materialized):
            raise ValueError("raw_scores must contain real-valued scores, not complex values.")
        return _ORIGINAL_AS_CLASS_SCORE_MATRIX(materialized, classes, n_samples=n_samples)

    def class_score_matrix(
        model: Any,
        features: Sequence[Sequence[float]] | np.ndarray,
        *,
        classes: Sequence | np.ndarray | None = None,
        fallback_labels: Sequence | np.ndarray | None = None,
        score_methods: Sequence[str] = ("decision_function", "predict_proba"),
        predict_fallback: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        return _ORIGINAL_CLASS_SCORE_MATRIX(
            model,
            _class_score_feature_matrix(features),
            classes=classes,
            fallback_labels=fallback_labels,
            score_methods=score_methods,
            predict_fallback=predict_fallback,
        )

    predict_window_model.__name__ = original_predict_window_model.__name__
    predict_window_model.__doc__ = original_predict_window_model.__doc__
    score_windowed_decoding.__name__ = original_score_windowed_decoding.__name__
    score_windowed_decoding.__doc__ = original_score_windowed_decoding.__doc__
    permutation_score_curves.__name__ = original_permutation_score_curves.__name__
    permutation_score_curves.__doc__ = original_permutation_score_curves.__doc__
    permutation_p_from_accuracy.__name__ = original_permutation_p_from_accuracy.__name__
    permutation_p_from_accuracy.__doc__ = original_permutation_p_from_accuracy.__doc__
    cross_validate_feature_decoding.__name__ = original_cross_validate_feature_decoding.__name__
    cross_validate_feature_decoding.__doc__ = original_cross_validate_feature_decoding.__doc__
    as_class_score_matrix.__name__ = _ORIGINAL_AS_CLASS_SCORE_MATRIX.__name__
    as_class_score_matrix.__doc__ = _ORIGINAL_AS_CLASS_SCORE_MATRIX.__doc__
    class_score_matrix.__name__ = _ORIGINAL_CLASS_SCORE_MATRIX.__name__
    class_score_matrix.__doc__ = _ORIGINAL_CLASS_SCORE_MATRIX.__doc__
    windowed.predict_window_model = predict_window_model
    windowed.score_windowed_decoding = score_windowed_decoding
    windowed.permutation_score_curves = permutation_score_curves
    windowed.permutation_p_from_accuracy = permutation_p_from_accuracy
    transfer.predict_window_model = predict_window_model
    transfer.score_windowed_decoding = score_windowed_decoding
    transfer.cross_validate_feature_decoding = cross_validate_feature_decoding
    class_scores.as_class_score_matrix = as_class_score_matrix
    class_scores.class_score_matrix = class_score_matrix
    _INSTALLED = True
