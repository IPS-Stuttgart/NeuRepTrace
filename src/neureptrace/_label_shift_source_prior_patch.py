"""Validate label-shift source priors, probability matrices, and scalar controls."""

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
_SCALAR_ERROR_SUFFIXES = {
    "_positive_int": "must be a positive integer.",
    "_positive_float": "must be positive and finite.",
    "_nonnegative_float": "must be finite and non-negative.",
}


def _source_prior_value_for_class(label_shift: Any, source_prior: Mapping[Any, float], class_label: Any) -> tuple[bool, Any]:
    """Return a mapping prior value using NeuRepTrace object-label equality."""

    for prior_label, prior_value in source_prior.items():
        if values_equal(prior_label, class_label) or label_shift._object_equal(prior_label, class_label):
            return True, prior_value
    return False, None


def _source_prior_values_for_classes(label_shift: Any, source_prior: Mapping[Any, float], classes: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    """Resolve mapping source-prior values for class labels that may be NaN/composite objects."""

    values: list[Any] = []
    missing: list[Any] = []
    for class_label in classes:
        found, prior_value = _source_prior_value_for_class(label_shift, source_prior, class_label)
        if found:
            values.append(prior_value)
        else:
            missing.append(class_label)
    return values, missing


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


def install() -> None:
    """Reject malformed label-shift priors, probabilities, labels, and scalar config values."""

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
                values, missing = _source_prior_values_for_classes(label_shift, source_prior, classes)
                if missing:
                    raise ValueError(
                        "source_prior mapping must provide a prior for every class; "
                        f"missing class label(s): {_format_label_preview(missing)}."
                    )
                return label_shift._prior_vector(
                    np.asarray(values, dtype=float),
                    n_classes=n_classes,
                    name="source_prior",
                    epsilon=epsilon,
                )
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
            return original_probability_matrix(
                values,
                name=name,
                epsilon=epsilon,
                expected_classes=expected_classes,
            )

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


__all__ = ["install"]
