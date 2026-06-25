"""Keep degenerate null-fallback CV predictions inside the observed label space."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from neureptrace.decoding.generative_augmentation import GenerativeAugmentationConfig

_INSTALLED = False
_ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING = None


def _sorted_label_values(labels: np.ndarray) -> np.ndarray:
    """Return the deterministic label order used by the transfer decoders."""

    values = sorted(np.unique(labels))
    if labels.dtype == object:
        return np.asarray(values, dtype=object)
    return np.asarray(values)


def _label_space_mask(predictions: np.ndarray, label_values: np.ndarray) -> np.ndarray:
    """Return a mask of predictions that belong to the observed non-null labels."""

    valid = np.zeros(predictions.shape, dtype=bool)
    for label in label_values:
        valid |= predictions == label
    return valid


def _all_same(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    flat = values.reshape(-1)
    return bool(np.all(flat == flat[0]))


def _repair_degenerate_out_of_space_predictions(predictions: np.ndarray, label_values: np.ndarray) -> np.ndarray:
    """Replace the all-null fallback label with a valid observed label.

    ``replace_null_class_predictions`` uses its scalar ``fallback_label`` only
    when every fold predicts the artificial null class.  Historically that
    default was label 1, which can be outside the observed label set.  Only this
    degenerate all-out-of-label-space case is repaired here; mixed valid/invalid
    model outputs are left untouched rather than silently masking a classifier
    error.
    """

    repaired = np.asarray(predictions).copy()
    if repaired.size == 0 or label_values.size == 0:
        return repaired

    valid = _label_space_mask(repaired, label_values)
    if bool(valid.any()) or not _all_same(repaired):
        return repaired

    fallback = label_values[0]
    if repaired.dtype == object:
        repaired[~valid] = fallback
        return repaired

    try:
        fallback = np.asarray([fallback], dtype=repaired.dtype)[0]
    except (TypeError, ValueError, OverflowError):
        repaired = repaired.astype(object)
    repaired[~valid] = fallback
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
            _sorted_label_values(label_vector),
        )
        accuracy = float(np.mean(label_vector == predictions)) if len(label_vector) else np.nan
        return transfer.CrossValidationResult(accuracy=accuracy, predictions=predictions, fold_ids=result.fold_ids)

    _cross_validate_feature_decoding.__name__ = _ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING.__name__
    _cross_validate_feature_decoding.__doc__ = _ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING.__doc__
    transfer.cross_validate_feature_decoding = _cross_validate_feature_decoding
    _INSTALLED = True
