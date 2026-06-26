"""Validate label-shift source-prior mappings against the class order."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_label_shift_source_prior_patch_installed"


def _missing_class_labels(source_prior: Mapping[Any, float], classes: Sequence[Any]) -> list[Any]:
    return [class_label for class_label in classes if class_label not in source_prior]


def _format_label_preview(labels: Sequence[Any], *, limit: int = 5) -> str:
    preview = ", ".join(repr(label) for label in labels[:limit])
    return f"{preview}, ..." if len(labels) > limit else preview


def install() -> None:
    """Reject source-prior mappings that omit one or more requested classes."""

    label_shift = importlib.import_module("neureptrace.decoding.label_shift")
    original_resolve = label_shift._resolve_source_prior
    if getattr(original_resolve, _PATCH_MARKER, False):
        return

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

    setattr(_resolve_source_prior, _PATCH_MARKER, True)
    _resolve_source_prior.__wrapped__ = original_resolve
    label_shift._resolve_source_prior = _resolve_source_prior


__all__ = ["install"]
