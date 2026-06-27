"""Runtime patch for stricter transfer-component-analysis validation."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_BOOL_TYPES = (bool, np.bool_)
_TRUE_BOOL_ALIASES = {"1", "true", "yes", "y", "on", "enable", "enabled"}
_FALSE_BOOL_ALIASES = {"0", "false", "no", "n", "off", "disable", "disabled"}
_GAMMA_ERROR = "gamma must be positive, 'median', or None, not a boolean value."
_COMPONENTS_ERROR = "n_components must be a positive integer, 'all', or infinity, not a boolean value."
_SAMPLE_WEIGHT_ERROR = "sample_weight must be a one-dimensional numeric weight vector, not a boolean mask or matrix."


def _normalize_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, _BOOL_TYPES):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower().replace("-", "_")
        if text in _TRUE_BOOL_ALIASES:
            return True
        if text in _FALSE_BOOL_ALIASES:
            return False
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be a boolean value.")


def _normalize_bool_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(kwargs)
    for name in ("standardize", "normalize_components"):
        if name in normalized:
            normalized[name] = _normalize_bool(normalized[name], name=name)
    return normalized


def _reject_bool_gamma(gamma: Any) -> None:
    if isinstance(gamma, _BOOL_TYPES):
        raise ValueError(_GAMMA_ERROR)


def _reject_bool_components(n_components: Any) -> None:
    if isinstance(n_components, _BOOL_TYPES):
        raise ValueError(_COMPONENTS_ERROR)


def _contains_bool(values: np.ndarray) -> bool:
    return any(isinstance(value, _BOOL_TYPES) for value in values.reshape(-1))


def _sample_weight_vector(sample_weight: Any, *, expected_length: int) -> np.ndarray:
    raw = np.asarray(sample_weight, dtype=object)
    if raw.ndim == 0:
        raise ValueError(_SAMPLE_WEIGHT_ERROR)
    if raw.ndim == 1:
        values = raw
    elif raw.ndim == 2 and raw.shape[1] == 1:
        values = raw[:, 0]
    elif raw.ndim == 2 and raw.shape[0] == 1:
        values = raw.reshape(-1)
    else:
        raise ValueError(_SAMPLE_WEIGHT_ERROR)
    if values.shape[0] != expected_length:
        raise ValueError(f"sample_weight must contain one value per source row: {values.shape[0]} != {expected_length}.")
    if _contains_bool(values):
        raise ValueError(_SAMPLE_WEIGHT_ERROR)
    weights = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("sample_weight must contain finite non-negative values.")
    return weights


def install() -> None:
    """Install strict validation for TCA hyperparameters, boolean flags, and source weights."""

    import neureptrace.decoding.transfer_component_analysis as tca

    if getattr(tca, "_transfer_component_analysis_validation_patched", False):
        return

    original_transfer_component_analysis_features = tca.transfer_component_analysis_features
    original_fit_tca_transfer_classifier = tca.fit_tca_transfer_classifier
    original_resolve_gamma = tca._resolve_gamma
    original_effective_components = tca._effective_components

    @wraps(original_transfer_component_analysis_features)
    def transfer_component_analysis_features(*args: Any, **kwargs: Any):
        normalized_kwargs = _normalize_bool_kwargs(kwargs)
        if "gamma" in normalized_kwargs:
            _reject_bool_gamma(normalized_kwargs["gamma"])
        if "n_components" in normalized_kwargs:
            _reject_bool_components(normalized_kwargs["n_components"])
        return original_transfer_component_analysis_features(*args, **normalized_kwargs)

    @wraps(original_fit_tca_transfer_classifier)
    def fit_tca_transfer_classifier(*args: Any, sample_weight: Any = None, **kwargs: Any):
        normalized_kwargs = _normalize_bool_kwargs(kwargs)
        if "gamma" in normalized_kwargs:
            _reject_bool_gamma(normalized_kwargs["gamma"])
        if "n_components" in normalized_kwargs:
            _reject_bool_components(normalized_kwargs["n_components"])
        if sample_weight is not None and not args and "source_features" in normalized_kwargs:
            n_source = tca._feature_matrix(normalized_kwargs["source_features"], name="source_features").shape[0]
            sample_weight = _sample_weight_vector(sample_weight, expected_length=n_source)
        return original_fit_tca_transfer_classifier(*args, sample_weight=sample_weight, **normalized_kwargs)

    @wraps(original_resolve_gamma)
    def _resolve_gamma(matrix: np.ndarray, *, gamma: Any, kernel: str, epsilon: float):
        _reject_bool_gamma(gamma)
        return original_resolve_gamma(matrix, gamma=gamma, kernel=kernel, epsilon=epsilon)

    @wraps(original_effective_components)
    def _effective_components(value: Any, *, max_components: int) -> int:
        _reject_bool_components(value)
        return original_effective_components(value, max_components=max_components)

    tca.transfer_component_analysis_features = transfer_component_analysis_features
    tca.fit_tca_transfer_classifier = fit_tca_transfer_classifier
    tca._resolve_gamma = _resolve_gamma
    tca._effective_components = _effective_components
    tca._transfer_component_analysis_validation_patched = True


__all__ = ["install"]
