"""Preserve non-numeric labels in feature-level cross-validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from neureptrace.decoding.generative_augmentation import GenerativeAugmentationConfig

_INSTALLED = False
_ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING = None


def _needs_object_predictions(labels: np.ndarray) -> bool:
    """Return true when labels cannot be stored losslessly in a float array."""

    return not np.issubdtype(labels.dtype, np.number)


def _coerced_null_label(null_label: object, labels: np.ndarray) -> object:
    """Match append_null_class_features' label dtype coercion for null rows."""

    return np.asarray([null_label], dtype=labels.dtype)[0]


def install() -> None:
    """Install the object-label-safe cross-validation wrapper once."""

    global _INSTALLED, _ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING
    if _INSTALLED:
        return

    from neureptrace.decoding import transfer

    _ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING = transfer.cross_validate_feature_decoding

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
        class_labels = np.asarray(sorted(np.unique(label_vector)), dtype=object)
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
                fold_predictions = transfer._one_vs_rest_predictions(
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

        predictions = transfer.replace_null_class_predictions(predictions, null_label=null_label_value)
        accuracy = float(np.mean(label_vector == predictions)) if len(label_vector) else np.nan
        return transfer.CrossValidationResult(accuracy=accuracy, predictions=predictions, fold_ids=fold_ids)

    _cross_validate_feature_decoding.__name__ = _ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING.__name__
    _cross_validate_feature_decoding.__doc__ = _ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING.__doc__
    transfer.cross_validate_feature_decoding = _cross_validate_feature_decoding
    _INSTALLED = True
