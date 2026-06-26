"""Runtime patch for stricter transfer-component validation."""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import Any

import numpy as np

_BOOL_TYPES = (bool, np.bool_)
_TRUE_BOOL_ALIASES = {"1", "true", "yes", "y", "on", "enable", "enabled"}
_FALSE_BOOL_ALIASES = {"0", "false", "no", "n", "off", "disable", "disabled"}
_GAMMA_ERROR = "gamma must be positive, 'scale', 'auto', or 'median', not a boolean value."
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


def _reject_bool_gamma(gamma: Any) -> None:
    if isinstance(gamma, _BOOL_TYPES):
        raise ValueError(_GAMMA_ERROR)


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
    """Install stricter validation for transfer-component configuration and source weights."""

    import neureptrace.decoding.transfer_components as transfer_components

    if getattr(transfer_components, "_transfer_components_validation_patched", False):
        return

    original_transfer_component_config = transfer_components.transfer_component_config
    original_coerce_config = transfer_components._coerce_config
    original_resolve_gamma = transfer_components._resolve_gamma
    original_fit_transfer_component_classifier = transfer_components.fit_transfer_component_classifier

    @wraps(original_transfer_component_config)
    def transfer_component_config(*args: Any, **kwargs: Any):
        if args:
            return original_transfer_component_config(*args, **kwargs)
        normalized_kwargs = dict(kwargs)
        if "center_kernel" in normalized_kwargs:
            normalized_kwargs["center_kernel"] = _normalize_bool(normalized_kwargs["center_kernel"], name="center_kernel")
        _reject_bool_gamma(normalized_kwargs.get("gamma", None))
        return original_transfer_component_config(**normalized_kwargs)

    @wraps(original_coerce_config)
    def _coerce_config(config: Any):
        cfg = original_coerce_config(config)
        _reject_bool_gamma(cfg.gamma)
        return replace(cfg, center_kernel=_normalize_bool(cfg.center_kernel, name="center_kernel"))

    @wraps(original_resolve_gamma)
    def _resolve_gamma(gamma: Any, *, features: np.ndarray, squared_distances: np.ndarray) -> float:
        _reject_bool_gamma(gamma)
        return original_resolve_gamma(gamma, features=features, squared_distances=squared_distances)

    @wraps(original_fit_transfer_component_classifier)
    def fit_transfer_component_classifier(*args: Any, sample_weight: Any = None, **kwargs: Any):
        if sample_weight is not None and not args and "source_features" in kwargs:
            n_source = transfer_components._feature_matrix(kwargs["source_features"], name="source_features").shape[0]
            sample_weight = _sample_weight_vector(sample_weight, expected_length=n_source)
        return original_fit_transfer_component_classifier(*args, sample_weight=sample_weight, **kwargs)

    transfer_components.transfer_component_config = transfer_component_config
    transfer_components._coerce_config = _coerce_config
    transfer_components._resolve_gamma = _resolve_gamma
    transfer_components.fit_transfer_component_classifier = fit_transfer_component_classifier
    transfer_components._transfer_components_validation_patched = True


__all__ = ["install"]
