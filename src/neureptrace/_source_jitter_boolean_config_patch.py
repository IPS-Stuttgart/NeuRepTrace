"""Normalize Source Feature Jitter boolean config values."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_jitter_boolean_config_patch_installed"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}


def _bool_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a boolean value.")


def _normalize_bool(value: Any, *, name: str) -> bool:
    """Return a strict bool for YAML/CLI-style jitter config values."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
        raise _bool_error(name)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _bool_error(name)
        return _normalize_bool(value.item(), name=name)
    if isinstance(value, (int, np.integer)):
        if int(value) in {0, 1}:
            return bool(value)
        raise _bool_error(name)
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value) in {0.0, 1.0}:
            return bool(value)
        raise _bool_error(name)
    raise _bool_error(name)


def install() -> None:
    """Install numeric/scalar boolean normalization for Source Feature Jitter."""

    source_jitter = importlib.import_module("neureptrace.decoding.source_jitter")

    original_config = source_jitter.source_feature_jitter_config
    if getattr(original_config, _PATCH_MARKER, False):
        return

    @wraps(original_config)
    def source_feature_jitter_config(
        *,
        synthetic_per_class: int | str = 0,
        noise_scale: float | str = source_jitter.DEFAULT_NOISE_SCALE,
        scale_mode: str | None = "global",
        preserve_original: Any = True,
        random_state: int | str | None = 13,
        epsilon: float | str = source_jitter.DEFAULT_EPSILON,
    ):
        return original_config(
            synthetic_per_class=synthetic_per_class,
            noise_scale=noise_scale,
            scale_mode=scale_mode,
            preserve_original=_normalize_bool(preserve_original, name="preserve_original"),
            random_state=random_state,
            epsilon=epsilon,
        )

    setattr(source_feature_jitter_config, _PATCH_MARKER, True)
    source_jitter.source_feature_jitter_config = source_feature_jitter_config


__all__ = ["install"]
