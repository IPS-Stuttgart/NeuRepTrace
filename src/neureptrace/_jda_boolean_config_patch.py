"""Normalize Joint Distribution Adaptation boolean config values."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_jda_boolean_config_patch_installed"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}


def _bool_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a boolean value.")


def _normalize_bool(value: Any, *, name: str) -> bool:
    """Return a real bool while rejecting ambiguous truthy/falsy objects."""

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
    """Install strict JDA boolean option normalization."""

    jda = importlib.import_module("neureptrace.decoding.joint_distribution_adaptation")
    original_config = jda.joint_distribution_adaptation_config
    if getattr(original_config, _PATCH_MARKER, False):
        return

    @wraps(original_config)
    def joint_distribution_adaptation_config(
        *,
        method: str | None = "jda",
        n_components: int | str | None = 16,
        max_iterations: int | str = 10,
        conditional_weight: float | str = 1.0,
        regularization: float | str = 1e-3,
        eigen_ridge: float | str = 1e-6,
        temperature: float | str = 1.0,
        standardize: Any = True,
        normalize_latent: Any = False,
    ):
        return original_config(
            method=method,
            n_components=n_components,
            max_iterations=max_iterations,
            conditional_weight=conditional_weight,
            regularization=regularization,
            eigen_ridge=eigen_ridge,
            temperature=temperature,
            standardize=_normalize_bool(standardize, name="standardize"),
            normalize_latent=_normalize_bool(normalize_latent, name="normalize_latent"),
        )

    setattr(joint_distribution_adaptation_config, _PATCH_MARKER, True)
    jda.joint_distribution_adaptation_config = joint_distribution_adaptation_config


__all__ = ["install"]
