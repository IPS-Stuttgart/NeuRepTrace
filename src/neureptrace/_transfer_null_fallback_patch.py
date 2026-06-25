"""Keep transfer CV fallback predictions inside the observed label space."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from neureptrace._object_label_utils import assign_masked, label_accuracy, values_equal
from neureptrace.decoding.generative_augmentation import GenerativeAugmentationConfig

_INSTALLED = False
_ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING = None


def _object_vector(values: Iterable[object]) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _ordered_unique(values: Sequence | np.ndarray) -> np.ndarray:
    unique: list[object] = []
    for value in values:
        if not any(values_equal(value, existing) for existing in unique):
            unique.append(value)
    return _object_vector(unique)


def _observed_label_values(labels: np.ndarray) -> np.ndarray:
    if np.issubdtype(labels.dtype, np.number):
        return np.asarray(sorted(np.unique(labels)))
    return _ordered_unique(labels)


def _label_space_mask(predictions: np.ndarray, label_values: np.ndarray) -> np.ndarray:
    return np.asarray([any(values_equal(prediction, label) for label in label_values) for prediction in predictions], dtype=bool)


def _all_same(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    flat = values.reshape(-1)
    first = flat[0]
    return all(values_equal(value, first) for value in flat)


def _repair_degenerate_out_of_space_predictions(predictions: np.ndarray, label_values: np.ndarray) -> np.ndarray:
    repaired = np.asarray(predictions).copy()
    if repaired.size == 0 or label_values.size == 0:
        return repaired

    valid = _label_space_mask(repaired, label_values)
    if bool(valid.any()) or not _all_same(repaired):
        return repaired

    fallback = label_values[0]
    if repaired.dtype != object:
        try:
            fallback = np.asarray([fallback], dtype=repaired.dtype)[0]
        except (TypeError, ValueError, OverflowError):
            repaired = repaired.astype(object)
    assign_masked(repaired, ~valid, fallback)
    return repaired


def install() -> None:
    """Install the cross-validation label-space fallback wrapper once."""

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
        result = _ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING(
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
        predictions = _repair_degenerate_out_of_space_predictions(
            result.predictions,
            _observed_label_values(label_vector),
        )
        accuracy = label_accuracy(label_vector, predictions)
        return transfer.CrossValidationResult(accuracy=accuracy, predictions=predictions, fold_ids=result.fold_ids)

    _cross_validate_feature_decoding.__name__ = _ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING.__name__
    _cross_validate_feature_decoding.__doc__ = _ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING.__doc__
    transfer.cross_validate_feature_decoding = _cross_validate_feature_decoding
    _INSTALLED = True
