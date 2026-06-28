"""Reject unsupported class/domain weighting options and guard small torch validation splits."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_torch_weight_validation_patch_installed"
_ROW_STRATIFIED_FALLBACK_CLASSES = frozenset(
    {
        "TorchMLPClassifier",
        "TorchDANNClassifier",
        "TorchCDANClassifier",
    }
)


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


def _small_stratified_holdout(labels: Any, fraction_value: Any) -> bool:
    fraction = _valid_fraction(fraction_value)
    if fraction is None:
        return False
    labels_array = np.asarray(labels).reshape(-1)
    if labels_array.size < 2:
        return True
    classes, counts = np.unique(labels_array, return_counts=True)
    if classes.shape[0] < 2 or counts.min() < 2:
        return True
    holdout_count = int(np.ceil(labels_array.size * fraction))
    return holdout_count < classes.shape[0] or labels_array.size - holdout_count < classes.shape[0]


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
        needs_training_loss_fallback = (
            class_object.__name__ in _ROW_STRATIFIED_FALLBACK_CLASSES
            and labels is not None
            and _small_stratified_holdout(labels, validation_fraction)
        )
        if not needs_training_loss_fallback:
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
