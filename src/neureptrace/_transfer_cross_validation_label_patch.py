"""Preserve non-numeric labels in feature-level cross-validation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from neureptrace.decoding.generative_augmentation import GenerativeAugmentationConfig

_INSTALLED = False
_ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING = None
_ORIGINAL_LABEL_VECTOR = None


def _object_vector(values: Iterable[object]) -> np.ndarray:
    """Build a one-dimensional object array without expanding tuple/list values."""

    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _numeric_array(labels: Sequence | np.ndarray) -> np.ndarray | None:
    """Return a numeric ndarray view when NumPy can represent labels numerically."""

    try:
        array = np.asarray(labels)
    except ValueError:
        return None
    return array if np.issubdtype(array.dtype, np.number) else None


def _atomic_label_vector(labels: Sequence | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    """Normalize labels while preserving one tuple/list-valued label per row."""

    numeric = _numeric_array(labels)
    array = np.asarray(labels, dtype=object)
    if array.ndim == 0:
        vector = _object_vector([array.item()])
    elif array.ndim == 1:
        vector = numeric.reshape(-1) if numeric is not None and numeric.ndim == 1 else array.reshape(-1)
    elif 1 in array.shape:
        vector = numeric.reshape(-1) if numeric is not None and 1 in numeric.shape else array.reshape(-1)
    else:
        rows = array.reshape(array.shape[0], -1)
        vector = _object_vector(tuple(row.tolist()) for row in rows)
    if len(vector) != expected_length:
        raise ValueError(f"{name} length must match feature rows: {len(vector)} != {expected_length}.")
    return vector


def _values_equal(left: object, right: object) -> bool:
    """Compare possibly composite labels without leaking array-valued equality."""

    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    try:
        return bool(equal)
    except (TypeError, ValueError):
        return False


def _label_equal_mask(values: Sequence | np.ndarray, label: object) -> np.ndarray:
    """Return an equality mask that is safe for tuple/list-valued object labels."""

    array = np.asarray(values, dtype=object)
    return np.asarray([_values_equal(value, label) for value in array], dtype=bool)


def _ordered_unique(values: Sequence | np.ndarray) -> np.ndarray:
    """Return stable unique values without sorting heterogeneous object labels."""

    unique: list[object] = []
    for value in values:
        if not any(_values_equal(value, existing) for existing in unique):
            unique.append(value)
    return _object_vector(unique)


def _label_counts(values: Sequence | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return stable unique values and counts for arbitrary object labels."""

    unique: list[object] = []
    counts: list[int] = []
    for value in values:
        for index, existing in enumerate(unique):
            if _values_equal(value, existing):
                counts[index] += 1
                break
        else:
            unique.append(value)
            counts.append(1)
    return _object_vector(unique), np.asarray(counts, dtype=int)


def _assign_masked(array: np.ndarray, mask: np.ndarray, value: object) -> None:
    """Assign one possibly composite scalar value to every true mask position."""

    if array.dtype == object:
        for index in np.flatnonzero(mask):
            array[index] = value
        return
    array[mask] = value


def _replace_null_class_predictions(predictions: Sequence | np.ndarray, *, null_label: object = 0, fallback_label: object = 1) -> np.ndarray:
    """Replace null predictions without broadcasting tuple/list fallback labels."""

    repaired = np.asarray(predictions).copy()
    null_mask = _label_equal_mask(repaired, null_label)
    if not np.any(null_mask):
        return repaired
    non_null = repaired[~null_mask]
    if len(non_null) == 0:
        _assign_masked(repaired, null_mask, fallback_label)
        return repaired
    nonzero_labels, counts = _label_counts(non_null)
    _assign_masked(repaired, null_mask, nonzero_labels[int(np.argmin(counts))])
    return repaired


def _label_accuracy(labels: Sequence | np.ndarray, predictions: Sequence | np.ndarray) -> float:
    """Return mean equality for labels that may be tuple/list-valued objects."""

    if len(labels) == 0:
        return np.nan
    return float(np.mean([_values_equal(label, prediction) for label, prediction in zip(labels, predictions, strict=True)]))


def _needs_object_predictions(labels: np.ndarray) -> bool:
    """Return true when labels cannot be stored losslessly in a float array."""

    return not np.issubdtype(labels.dtype, np.number)


def _coerced_null_label(null_label: object, labels: np.ndarray) -> object:
    """Match append_null_class_features' label dtype coercion for null rows."""

    if labels.dtype == object:
        return null_label
    return np.asarray([null_label], dtype=labels.dtype)[0]


def install() -> None:
    """Install the object-label-safe cross-validation wrapper once."""

    global _INSTALLED, _ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING, _ORIGINAL_LABEL_VECTOR
    if _INSTALLED:
        return

    from neureptrace.decoding import transfer

    _ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING = transfer.cross_validate_feature_decoding
    _ORIGINAL_LABEL_VECTOR = transfer._label_vector
    transfer._label_vector = _atomic_label_vector

    def _one_vs_rest_predictions_object(
        train_features: np.ndarray,
        train_labels: np.ndarray,
        test_features: np.ndarray,
        class_labels: np.ndarray,
        *,
        classifier: str,
        classifier_param: Any,
        components_pca: int | float,
        random_state: int | None,
    ) -> np.ndarray:
        all_scores = np.zeros((test_features.shape[0], len(class_labels)))
        for class_index, class_label in enumerate(class_labels):
            binary_bundle = transfer.fit_window_model(
                train_features,
                _label_equal_mask(train_labels, class_label),
                fit_model=lambda features, labels: transfer._fit_binary_model(
                    features,
                    labels,
                    classifier=classifier,
                    classifier_param=classifier_param,
                    random_state=random_state,
                ),
                components_pca=components_pca,
            )
            transformed_test = transfer.transform_window_features(binary_bundle, test_features)
            if classifier in ("lasso", "svm-binary", "binary-svm"):
                all_scores[:, class_index] = transfer.positive_class_score(binary_bundle.model, transformed_test)
            else:
                all_scores[:, class_index] = binary_bundle.model.predict(transformed_test)
        return class_labels[np.argmax(all_scores, axis=1)]

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
    def _cross_validate_feature_decoding(
        stimulus_features: Sequence[Sequence[float]] | np.ndarray,
        labels: Sequence | np.ndarray,
        *,
        null_features: Sequence[Sequence[float]] | np.ndarray | None = None,
        n_folds: int = 10,
        classifier: str = "multiclass-svm",
        classifier_param: Any = 0.5,
        components_pca: int | float = float("inf"),
        random_state: int | None = None,
        fit_model: Callable[[np.ndarray, np.ndarray], Any] | None = None,
        null_label: int | float = 0,
        generative_augmentation: GenerativeAugmentationConfig | Mapping[str, Any] | None = None,
    ) -> transfer.CrossValidationResult:
        stimulus_features_array = transfer._feature_matrix(stimulus_features, name="stimulus_features")
        label_vector = transfer._label_vector(labels, expected_length=stimulus_features_array.shape[0], name="labels")
        if not _needs_object_predictions(label_vector):
            return _ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING(
                stimulus_features_array,
                label_vector,
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

        n_trials = len(label_vector)
        fold_ids = transfer.sequential_fold_ids(n_trials, n_folds)
        features, augmented_labels = transfer.append_null_class_features(stimulus_features_array, label_vector, null_features, null_label=null_label)
        null_label_value = _coerced_null_label(null_label, augmented_labels)
        augmented_folds = fold_ids
        if null_features is not None:
            null_features_array = transfer._feature_matrix(null_features, name="null_features")
            if null_features_array.shape[0] != n_trials:
                raise ValueError("null_features must contain one row per stimulus trial for fold augmentation.")
            augmented_folds = np.concatenate([fold_ids, fold_ids])

        predictions = np.empty(n_trials, dtype=object)
        predictions[:] = None
        class_labels = _ordered_unique(label_vector)
        fallback_label = class_labels[0] if len(class_labels) else null_label_value
        for fold in range(1, n_folds + 1):
            train_mask = augmented_folds != fold
            test_mask = (augmented_folds == fold) & (augmented_labels != null_label_value)
            if not np.any(test_mask):
                continue
            train_features = features[train_mask]
            train_labels = augmented_labels[train_mask]
            train_features, train_labels = transfer._apply_generative_augmentation(train_features, train_labels, generative_augmentation=generative_augmentation)
            test_features = features[test_mask]

            if classifier in transfer.BINARY_ONE_VS_REST_CLASSIFIERS:
                fold_predictions = _one_vs_rest_predictions_object(
                    train_features,
                    train_labels,
                    test_features,
                    class_labels,
                    classifier=classifier,
                    classifier_param=classifier_param,
                    components_pca=components_pca,
                    random_state=random_state,
                )
            else:
                model_bundle = transfer.fit_window_model(
                    train_features,
                    train_labels,
                    fit_model=transfer._fit_model(classifier, classifier_param, random_state, fit_model),
                    components_pca=components_pca,
                )
                fold_predictions, _ = transfer.predict_window_model(model_bundle, test_features)
            predictions[fold_ids == fold] = fold_predictions

        predictions = _replace_null_class_predictions(predictions, null_label=null_label_value, fallback_label=fallback_label)
        accuracy = _label_accuracy(label_vector, predictions)
        return transfer.CrossValidationResult(accuracy=accuracy, predictions=predictions, fold_ids=fold_ids)

    _cross_validate_feature_decoding.__name__ = _ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING.__name__
    _cross_validate_feature_decoding.__doc__ = _ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING.__doc__
    transfer.cross_validate_feature_decoding = _cross_validate_feature_decoding
    _INSTALLED = True
