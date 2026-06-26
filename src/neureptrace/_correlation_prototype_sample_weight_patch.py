"""Runtime patch for weighted correlation-prototype class-support validation."""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_correlation_prototype_sample_weight_patch_installed"


def _as_python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _zero_weight_classes(labels: Sequence[object] | np.ndarray, sample_weight: Sequence[float] | np.ndarray) -> list[Any]:
    labels_array = np.asarray(labels).ravel()
    try:
        weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return []
    if weights.shape[0] != labels_array.shape[0]:
        return []
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        return []

    return [
        _as_python_scalar(class_label)
        for class_label in np.unique(labels_array)
        if float(weights[labels_array == class_label].sum()) <= 0.0
    ]


def install() -> None:
    """Reject weighted prototype fits that leave any class without support."""

    from neureptrace.decoding import classifiers

    classifier_type = classifiers.CorrelationPrototypeClassifier
    if getattr(classifier_type.fit, _PATCH_MARKER, False):
        return

    original_fit = classifier_type.fit

    @wraps(original_fit)
    def fit(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        labels: Sequence[object] | np.ndarray,
        sample_weight: Sequence[float] | np.ndarray | None = None,
    ):
        if sample_weight is not None:
            unsupported_classes = _zero_weight_classes(labels, sample_weight)
            if unsupported_classes:
                raise ValueError(
                    "sample_weight must assign positive total weight to every class; "
                    f"zero-weight classes: {unsupported_classes!r}."
                )
        return original_fit(self, features, labels, sample_weight=sample_weight)

    setattr(fit, _PATCH_MARKER, True)
    classifier_type.fit = fit


__all__ = ["install"]
