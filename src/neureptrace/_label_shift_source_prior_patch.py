"""Validate label-shift source priors and probability matrices."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

import numpy as np

_SOURCE_PRIOR_PATCH_MARKER = "_neureptrace_label_shift_source_prior_patch_installed"
_PROBABILITY_MATRIX_PATCH_MARKER = "_neureptrace_label_shift_probability_matrix_patch_installed"


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


def install() -> None:
    """Reject incomplete source-prior mappings and boolean probability matrices."""

    label_shift = importlib.import_module("neureptrace.decoding.label_shift")

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
            if isinstance(source_prior, Mapping):
                missing = _missing_class_labels(source_prior, classes)
                if missing:
                    raise ValueError(
                        "source_prior mapping must provide a prior for every class; "
                        f"missing class label(s): {_format_label_preview(missing)}."
                    )
                values = np.asarray([source_prior[class_label] for class_label in classes], dtype=float)
                return label_shift._prior_vector(values, n_classes=n_classes, name="source_prior", epsilon=epsilon)
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
    if getattr(original_probability_matrix, _PROBABILITY_MATRIX_PATCH_MARKER, False):
        return

    @wraps(original_probability_matrix)
    def _probability_matrix(values, *, name: str, epsilon: float | str, expected_classes: int | None = None):
        _reject_boolean_probability_values(values, name=name)
        return original_probability_matrix(values, name=name, epsilon=epsilon, expected_classes=expected_classes)

    setattr(_probability_matrix, _PROBABILITY_MATRIX_PATCH_MARKER, True)
    _probability_matrix.__wrapped__ = original_probability_matrix
    label_shift._probability_matrix = _probability_matrix


__all__ = ["install"]
