"""Reject unsupported class/domain weighting options in torch decoders."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_torch_weight_validation_patch_installed"


def _validate_weight_option(value: Any, *, name: str) -> None:
    """Accept only the documented torch decoder weighting modes."""

    if value is None or value == "balanced":
        return
    raise ValueError(f"{name} must be None or 'balanced'.")


def _valid_fraction(value: Any) -> float | None:
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(fraction) or fraction <= 0.0 or fraction >= 1.0:
        return None
    return fraction


def _labels_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any | None:
    if len(args) >= 2:
        return args[1]
    return kwargs.get("labels", kwargs.get("y"))


def _label_vector(labels: Any) -> np.ndarray:
    """Return one label object per sample, preserving composite labels."""

    labels_array = np.asarray(labels, dtype=object)
    if labels_array.ndim == 0:
        vector = np.empty(1, dtype=object)
        vector[0] = labels_array.item()
        return vector
    if labels_array.ndim == 1:
        return labels_array.reshape(-1)

    rows = labels_array.reshape(labels_array.shape[0], -1)
    if rows.shape[1] == 1:
        return rows[:, 0].reshape(-1)
    vector = np.empty(rows.shape[0], dtype=object)
    for index, row in enumerate(rows):
        vector[index] = tuple(row.tolist())
    return vector


def _values_equal(left: Any, right: Any) -> bool:
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    try:
        return bool(equal)
    except (TypeError, ValueError):
        return False


def _class_counts(labels: np.ndarray) -> list[int]:
    counts: list[int] = []
    classes: list[Any] = []
    for label in labels.tolist():
        for index, class_label in enumerate(classes):
            if _values_equal(label, class_label):
                counts[index] += 1
                break
        else:
            classes.append(label)
            counts.append(1)
    return counts


def _small_stratified_holdout(labels: Any, fraction_value: Any) -> bool:
    fraction = _valid_fraction(fraction_value)
    if fraction is None:
        return False
    labels_array = _label_vector(labels)
    if labels_array.size < 2:
        return True
    counts = _class_counts(labels_array)
    if len(counts) < 2 or min(counts) < 2:
        return True
    holdout_count = int(np.ceil(labels_array.size * fraction))
    return holdout_count < len(counts) or labels_array.size - holdout_count < len(counts)


def _install_fit_guard(class_object: type, *attribute_names: str) -> None:
    original_fit = class_object.fit
    if getattr(original_fit, _PATCH_MARKER, False):
        return

    @wraps(original_fit)
    def fit(self, *args, **kwargs):
        for attribute_name in attribute_names:
            _validate_weight_option(getattr(self, attribute_name, None), name=attribute_name)
        labels = _labels_from_call(args, kwargs)
        validation_fraction = getattr(self, "validation_fraction", None)
        if class_object.__name__ != "TorchMLPClassifier" or labels is None or not _small_stratified_holdout(labels, validation_fraction):
            return original_fit(self, *args, **kwargs)
        self.validation_fraction = 0.0
        try:
            return original_fit(self, *args, **kwargs)
        finally:
            self.validation_fraction = validation_fraction

    setattr(fit, _PATCH_MARKER, True)
    class_object.fit = fit


def install() -> None:
    """Install weight-option validation for torch-backed decoders."""

    decoding = importlib.import_module("neureptrace.decoding")
    _install_fit_guard(decoding.TorchMLPClassifier, "class_weight")

    dann = importlib.import_module("neureptrace.decoding.dann")
    _install_fit_guard(dann.TorchDANNClassifier, "class_weight")

    cdan = importlib.import_module("neureptrace.decoding.cdan")
    _install_fit_guard(cdan.TorchCDANClassifier, "class_weight")

    source_domain_generalization = importlib.import_module("neureptrace.decoding.source_domain_generalization")
    _install_fit_guard(
        source_domain_generalization.TorchSourceDomainGeneralizationClassifier,
        "class_weight",
        "domain_weight",
    )

    source_vrex = importlib.import_module("neureptrace.decoding.source_vrex")
    _install_fit_guard(source_vrex.TorchVRExClassifier, "class_weight")


__all__ = ["install"]
