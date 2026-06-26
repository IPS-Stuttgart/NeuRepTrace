"""Avoid conflating observed labels with the artificial transfer null label."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from neureptrace._object_label_utils import values_equal
from neureptrace.decoding.generative_augmentation import GenerativeAugmentationConfig

_INSTALLED = False
_ORIGINAL_APPEND_NULL_CLASS_FEATURES = None
_ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING = None


class _UnusedObjectNullLabel:
    """Sentinel that cannot compare equal to user-provided labels."""

    def __eq__(self, other: object) -> bool:
        return False

    def __ne__(self, other: object) -> bool:
        return True

    def __repr__(self) -> str:
        return "<unused-neureptrace-null-label>"


def _contains_label(labels: np.ndarray, label: object) -> bool:
    return any(values_equal(value, label) for value in labels.reshape(-1))


def _unused_numeric_null_label(labels: np.ndarray) -> object:
    for offset in range(1, 1_000_000):
        for candidate in (-offset, offset):
            if not _contains_label(labels, candidate):
                return candidate
    return _UnusedObjectNullLabel()


def _effective_null_label(labels: np.ndarray, null_label: object) -> object:
    if not _contains_label(labels, null_label):
        return null_label
    if np.issubdtype(labels.dtype, np.number):
        return _unused_numeric_null_label(labels)
    return _UnusedObjectNullLabel()


def _raise_ambiguous_null_label() -> None:
    raise ValueError(
        "null_label must not overlap observed labels when null_features are provided; "
        "pass a null_label outside the stimulus label space."
    )


def install() -> None:
    """Install transfer wrappers that keep stimulus and artificial-null labels disjoint."""

    global _INSTALLED, _ORIGINAL_APPEND_NULL_CLASS_FEATURES, _ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING
    if _INSTALLED:
        return

    from neureptrace.decoding import transfer

    _ORIGINAL_APPEND_NULL_CLASS_FEATURES = transfer.append_null_class_features
    _ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING = transfer.cross_validate_feature_decoding

    def _append_null_class_features(
        stimulus_features: Sequence[Sequence[float]] | np.ndarray,
        labels: Sequence | np.ndarray,
        null_features: Sequence[Sequence[float]] | np.ndarray | None = None,
        *,
        null_label: int | float = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        stimulus_features_array = transfer._feature_matrix(stimulus_features, name="stimulus_features")
        label_vector = transfer._label_vector(labels, expected_length=stimulus_features_array.shape[0], name="labels")
        if null_features is not None and _contains_label(label_vector, null_label):
            _raise_ambiguous_null_label()
        return _ORIGINAL_APPEND_NULL_CLASS_FEATURES(
            stimulus_features_array,
            label_vector,
            null_features,
            null_label=null_label,
        )

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
        if null_features is not None and _contains_label(label_vector, null_label):
            _raise_ambiguous_null_label()
        effective_null_label = null_label if null_features is not None else _effective_null_label(label_vector, null_label)
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
            null_label=effective_null_label,
            generative_augmentation=generative_augmentation,
        )

    _append_null_class_features.__name__ = _ORIGINAL_APPEND_NULL_CLASS_FEATURES.__name__
    _append_null_class_features.__doc__ = _ORIGINAL_APPEND_NULL_CLASS_FEATURES.__doc__
    _cross_validate_feature_decoding.__name__ = _ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING.__name__
    _cross_validate_feature_decoding.__doc__ = _ORIGINAL_CROSS_VALIDATE_FEATURE_DECODING.__doc__
    transfer.append_null_class_features = _append_null_class_features
    transfer.cross_validate_feature_decoding = _cross_validate_feature_decoding
    _INSTALLED = True
