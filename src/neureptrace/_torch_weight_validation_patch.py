"""Reject unsupported class/domain weighting options in torch decoders."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

_PATCH_MARKER = "_neureptrace_torch_weight_validation_patch_installed"


def _validate_weight_option(value: Any, *, name: str) -> None:
    """Accept only the documented torch decoder weighting modes."""

    if value is None or value == "balanced":
        return
    raise ValueError(f"{name} must be None or 'balanced'.")


def _install_fit_guard(class_object: type, *attribute_names: str) -> None:
    original_fit = class_object.fit
    if getattr(original_fit, _PATCH_MARKER, False):
        return

    @wraps(original_fit)
    def fit(self, *args, **kwargs):
        for attribute_name in attribute_names:
            _validate_weight_option(getattr(self, attribute_name, None), name=attribute_name)
        return original_fit(self, *args, **kwargs)

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
