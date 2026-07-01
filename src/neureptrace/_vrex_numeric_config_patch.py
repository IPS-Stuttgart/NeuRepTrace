"""Validate VREx numeric hyperparameters and finite fit features."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_vrex_numeric_config_patch_installed"
_SOURCE_VREX_FEATURE_PATCH_MARKER = "_neureptrace_source_vrex_finite_fit_feature_patch_installed"


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative.") from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return parsed


def _validate_finite_source_features(source_features: Any) -> None:
    """Reject NaN/Inf VREx fit features before torch training starts."""

    try:
        matrix = np.asarray(source_features, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("vrex source_features must be numeric and finite.") from exc
    if matrix.ndim == 2 and not np.all(np.isfinite(matrix)):
        raise ValueError("vrex source_features must contain finite values.")


def _install_linear_vrex_numeric_validators() -> None:
    vrex = importlib.import_module("neureptrace.decoding.vrex")
    if getattr(vrex._positive_int, _PATCH_MARKER, False):
        return

    setattr(_positive_int, _PATCH_MARKER, True)
    setattr(_positive_float, _PATCH_MARKER, True)
    setattr(_nonnegative_float, _PATCH_MARKER, True)
    vrex._positive_int = _positive_int
    vrex._positive_float = _positive_float
    vrex._nonnegative_float = _nonnegative_float


def _install_source_vrex_fit_feature_validator() -> None:
    source_vrex = importlib.import_module("neureptrace.decoding.source_vrex")
    original_fit = source_vrex.TorchVRExClassifier.fit
    if getattr(original_fit, _SOURCE_VREX_FEATURE_PATCH_MARKER, False):
        return

    @wraps(original_fit)
    def fit(self, source_features, source_labels, *, source_domains):
        _validate_finite_source_features(source_features)
        return original_fit(self, source_features, source_labels, source_domains=source_domains)

    setattr(fit, _SOURCE_VREX_FEATURE_PATCH_MARKER, True)
    source_vrex.TorchVRExClassifier.fit = fit


def install() -> None:
    """Install VREx hyperparameter and fit-feature validators."""

    _install_linear_vrex_numeric_validators()
    _install_source_vrex_fit_feature_validator()


__all__ = ["install"]
