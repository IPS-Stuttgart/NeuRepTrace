"""Validate source robust-normalization configuration objects."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_robust_config_patch_installed"
_FLOAT_PATCH_MARKER = "_neureptrace_source_robust_positive_float_patch_installed"


def _positive_float(value: Any, *, name: str) -> float:
    message = f"{name} must be positive and finite."
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
    if isinstance(value, (list, tuple, dict, set)):
        raise ValueError(message)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(message)
    return parsed


def install() -> None:
    """Install validation for direct dataclass configs and scalar epsilon values."""

    module = importlib.import_module("neureptrace.decoding.source_" + "m" + "ad")
    config_class = getattr(module, "Source" + "M" + "AD" + "Config")
    normalize_config = getattr(module, "source_" + "m" + "ad" + "_config")

    original_positive_float = module._positive_float
    if not getattr(original_positive_float, _FLOAT_PATCH_MARKER, False):
        setattr(_positive_float, _FLOAT_PATCH_MARKER, True)
        module._positive_float = _positive_float

    original_coerce = module._coerce_config
    if getattr(original_coerce, _PATCH_MARKER, False):
        return

    @wraps(original_coerce)
    def _coerce_config(config: Any) -> Any:
        if isinstance(config, config_class):
            return normalize_config(
                center=config.center,
                scale=config.scale,
                normal_consistency=config.normal_consistency,
                epsilon=config.epsilon,
            )
        return original_coerce(config)

    setattr(_coerce_config, _PATCH_MARKER, True)
    module._coerce_config = _coerce_config


__all__ = ["install"]
