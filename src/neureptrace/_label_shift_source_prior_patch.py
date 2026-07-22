"""Validate label-shift inputs and preserve source-prior label semantics."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

import numpy as np

from neureptrace._object_label_utils import values_equal

_SOURCE_PRIOR_PATCH_MARKER = "_neureptrace_label_shift_source_prior_patch_installed"
_PROBABILITY_MATRIX_PATCH_MARKER = "_neureptrace_label_shift_probability_matrix_patch_installed"
_SCALAR_CONFIG_PATCH_MARKER = "_neureptrace_label_shift_scalar_config_patch_installed"
_CLASSES_PATCH_MARKER = "_neureptrace_label_shift_classes_patch_installed"
_SOFT_CONFUSION_LABEL_PATCH_MARKER = "_neureptrace_label_shift_soft_confusion_label_patch_installed"
_SOURCE_PRIOR_LABEL_EQUALITY_PATCH_MARKER = "_neureptrace_source_prior_missing_label_kind_patch_installed"
_PRIOR_FROM_LABELS_PATCH_MARKER = "_neureptrace_prior_shift_class_coverage_patch_installed"
_SCALAR_ERROR_SUFFIXES = {
    "_positive_int": "must be a positive integer.",
    "_positive_float": "must be positive and finite.",
    "_nonnegative_float": "must be finite and non-negative.",
}


def _missing_class_labels(source_prior: Mapping[Any, float], classes: Sequence[Any]) -> list[Any]:
    return [class_label for class_label in classes if class_label not in source_prior]


def _format_label_preview(labels: Sequence[Any], *, limit: int = 5) -> str:
    preview = ", ".join(repr(label) for label in labels[:limit])
    return f"{preview}, ..." if len(labels) > limit else preview


def _contains_boolean_probability_values(values: Any) -> bool:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError):
        return False
    if np.issubdtype(raw.dtype, np.bool_):
        return True
    if raw.dtype == object:
        return any(isinstance(value, (bool, np.bool_)) for value in raw.reshape(-1))
    return False


def _reject_boolean_probability_values(values: Any, *, name: str) -> None:
    if _contains_boolean_probability_values(values):
        raise ValueError(f"{name} must be numeric probability values, not boolean indicators.")


def _class_label_unknowns(label_shift: Any, labels: Sequence[Any] | np.ndarray, classes: Sequence[Any] | np.ndarray) -> list[Any]:
    class_order = label_shift._object_vector(classes, name="classes")
    return [
        label
        for label in label_shift._unique_object_values(labels)
        if not any(label_shift._object_equal(label, class_label) for class_label in class_order.tolist())
    ]


def _validate_unique_classes(label_shift: Any, classes: Sequence[Any] | np.ndarray) -> None:
    class_order = label_shift._object_vector(classes, name="classes")
    if len(label_shift._unique_object_values(class_order)) != class_order.shape[0]:
        raise ValueError("classes must be unique.")


def _reject_unknown_source_labels(label_shift: Any, labels: Sequence[Any] | np.ndarray, classes: Sequence[Any]) -> None:
    unknown = _class_label_unknowns(label_shift, labels, classes)
    if unknown:
        raise ValueError(f"source labels contain labels absent from classes: {_format_label_preview(unknown)}.")


def _reject_unknown_validation_labels(label_shift: Any, labels: Sequence[Any] | np.ndarray, classes: Sequence[Any]) -> None:
    unknown = _class_label_unknowns(label_shift, labels, classes)
    if unknown:
        raise ValueError(f"source_validation_labels contain labels absent from classes: {_format_label_preview(unknown)}.")


def _array_scalar_error(name: str, suffix: str) -> ValueError:
    return ValueError(f"{name} {suffix}")


def _guarded_scalar_helper(original: Any, error_suffix: str):
    @wraps(original)
    def _helper(value: Any, *, name: str):
        if isinstance(value, np.ndarray):
            raise _array_scalar_error(name, error_suffix)
        return original(value, name=name)

    setattr(_helper, _SCALAR_CONFIG_PATCH_MARKER, True)
    _helper.__wrapped__ = original
    return _helper


def _install_scalar_config_guards(label_shift: Any) -> None:
    for helper_name, error_suffix in _SCALAR_ERROR_SUFFIXES.items():
        original = getattr(label_shift, helper_name)
        if getattr(original, _SCALAR_CONFIG_PATCH_MARKER, False):
            continue
        setattr(label_shift, helper_name, _guarded_scalar_helper(original, error_suffix))


def _install_source_prior_label_equality() -> None:
    """Keep distinct missing-value sentinel types as distinct source classes."""

    source_prior = importlib.import_module("neureptrace.decoding.source_prior")
    original = source_prior._object_equal
    if getattr(original, _SOURCE_PRIOR_LABEL_EQUALITY_PATCH_MARKER, False):
        return

    @wraps(original)
    def _object_equal(left: Any, right: Any) -> bool:
        if source_prior._is_missing_label_scalar(left) or source_prior._is_missing_label_scalar(right):
            return values_equal(left, right)
        return original(left, right)

    setattr(_object_equal, _SOURCE_PRIOR_LABEL_EQUALITY_PATCH_MARKER, True)
    _object_equal.__wrapped__ = original
    source_prior._object_equal = _object_equal


def _install_prior_from_labels_guard() -> None:
    """Require explicit prior-shift class lists to be unique and exhaustive."""

    prior_shift = importlib.import_module("neureptrace.decoding.prior_shift")
    original = prior_shift.prior_from_labels
    if getattr(original, _PRIOR_FROM_LABELS_PATCH_MARKER, False):
        return

    @wraps(original)
    def prior_from_labels(labels, classes=None, *, smoothing=0.0):
        if classes is None:
            return original(labels, classes=None, smoothing=smoothing)

        label_vector = prior_shift._object_vector(labels, name="labels")
        class_order = prior_shift._object_vector(classes, name="classes")
        if len(prior_shift._unique_values(class_order)) != class_order.shape[0]:
            raise ValueError("classes must be unique.")

        unknown = [
            label
            for label in prior_shift._unique_values(label_vector)
            if not any(prior_shift._object_equal(label, class_label) for class_label in class_order.tolist())
        ]
        if unknown:
            raise ValueError(f"labels contain labels absent from classes: {_format_label_preview(unknown)}.")

        return original(label_vector, classes=class_order, smoothing=smoothing)

    setattr(prior_from_labels, _PRIOR_FROM_LABELS_PATCH_MARKER, True)
    prior_from_labels.__wrapped__ = original
    prior_shift.prior_from_labels = prior_from_labels


def install() -> None:
    """Reject malformed label-shift inputs and install source-prior label guards."""

    label_shift = importlib.import_module("neureptrace.decoding.label_shift")

    original_resolve_classes = label_shift._resolve_classes
    if not getattr(original_resolve_classes, _CLASSES_PATCH_MARKER, False):

        @wraps(original_resolve_classes)
        def _resolve_classes(n_classes: int, classes, source_labels, source_validation_labels):
            class_order = original_resolve_classes(n_classes, classes, source_labels, source_validation_labels)
            _validate_unique_classes(label_shift, class_order)
            return class_order

        setattr(_resolve_classes, _CLASSES_PATCH_MARKER, True)
        _resolve_classes.__wrapped__ = original_resolve_classes
        label_shift._resolve_classes = _resolve_classes

    original_resolve = label_shift._resolve_source_prior
    if not getattr(original_resolve, _SOURCE_PRIOR_PATCH_MARKER, False):

        @wraps(original_resolve)
        def _resolve_source_prior(
            n_classes: int,
            *,
            source_prior,
            source_labels,
            source_validation_labels,
            classes: Sequence[Any],
            epsilon: float,
        ):
            _validate_unique_classes(label_shift, classes)
            if isinstance(source_prior, Mapping):
                missing = _missing_class_labels(source_prior, classes)
                if missing:
                    raise ValueError(
                        "source_prior mapping must provide a prior for every class; "
                        f"missing class label(s): {_format_label_preview(missing)}."
                    )
                values = np.asarray([source_prior[class_label] for class_label in classes], dtype=float)
                return label_shift._prior_vector(values, n_classes=n_classes, name="source_prior", epsilon=epsilon)
            labels = source_labels if source_labels is not None else source_validation_labels
            if labels is not None:
                _reject_unknown_source_labels(label_shift, labels, classes)
            return original_resolve(
                n_classes,
                source_prior=source_prior,
                source_labels=source_labels,
                source_validation_labels=source_validation_labels,
                classes=classes,
                epsilon=epsilon,
            )

        setattr(_resolve_source_prior, _SOURCE_PRIOR_PATCH_MARKER, True)
        _resolve_source_prior.__wrapped__ = original_resolve
        label_shift._resolve_source_prior = _resolve_source_prior

    original_probability_matrix = label_shift._probability_matrix
    if not getattr(original_probability_matrix, _PROBABILITY_MATRIX_PATCH_MARKER, False):

        @wraps(original_probability_matrix)
        def _probability_matrix(values, *, name: str, epsilon: float | str, expected_classes: int | None = None):
            _reject_boolean_probability_values(values, name=name)
            return original_probability_matrix(values, name=name, epsilon=epsilon, expected_classes=expected_classes)

        setattr(_probability_matrix, _PROBABILITY_MATRIX_PATCH_MARKER, True)
        _probability_matrix.__wrapped__ = original_probability_matrix
        label_shift._probability_matrix = _probability_matrix

    original_soft_confusion_matrix = label_shift.soft_confusion_matrix
    if not getattr(original_soft_confusion_matrix, _SOFT_CONFUSION_LABEL_PATCH_MARKER, False):

        @wraps(original_soft_confusion_matrix)
        def soft_confusion_matrix(source_validation_probabilities, source_validation_labels, *, classes, epsilon=1e-12):
            _validate_unique_classes(label_shift, classes)
            _reject_unknown_validation_labels(label_shift, source_validation_labels, classes)
            return original_soft_confusion_matrix(
                source_validation_probabilities,
                source_validation_labels,
                classes=classes,
                epsilon=epsilon,
            )

        setattr(soft_confusion_matrix, _SOFT_CONFUSION_LABEL_PATCH_MARKER, True)
        soft_confusion_matrix.__wrapped__ = original_soft_confusion_matrix
        label_shift.soft_confusion_matrix = soft_confusion_matrix

    _install_scalar_config_guards(label_shift)
    _install_source_prior_label_equality()
    _install_prior_from_labels_guard()


__all__ = ["install"]
