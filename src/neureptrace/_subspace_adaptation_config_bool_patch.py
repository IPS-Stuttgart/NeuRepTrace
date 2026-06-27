"""Normalize subspace-adaptation boolean configuration values."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_subspace_adaptation_config_bool_patch_installed"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}


def _bool_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a boolean value.")


def _normalize_bool(value: Any, *, name: str) -> bool:
    """Return a real bool while rejecting ambiguous CLI/YAML-style inputs."""

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
        integer = int(value)
        if integer in {0, 1}:
            return bool(integer)
        raise _bool_error(name)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and numeric in {0.0, 1.0}:
            return bool(numeric)
        raise _bool_error(name)
    raise _bool_error(name)


def install() -> None:
    """Install strict boolean normalization for subspace-adaptation config values."""

    subspace_adaptation = importlib.import_module("neureptrace.decoding.subspace_adaptation")
    original_config = subspace_adaptation.subspace_adaptation_config
    if getattr(original_config, _PATCH_MARKER, False):
        return

    @wraps(original_config)
    def subspace_adaptation_config(
        *,
        method: str | None = subspace_adaptation.DEFAULT_SUBSPACE_METHOD,
        n_components: int | str | None = subspace_adaptation.DEFAULT_SUBSPACE_COMPONENTS,
        regularization: float | str = subspace_adaptation.DEFAULT_SUBSPACE_REGULARIZATION,
        eigen_ridge: float | str = subspace_adaptation.DEFAULT_SUBSPACE_EIGEN_RIDGE,
        standardize: Any = True,
        class_balance_source: Any = False,
        normalize_latent: Any = False,
    ):
        return original_config(
            method=method,
            n_components=n_components,
            regularization=regularization,
            eigen_ridge=eigen_ridge,
            standardize=_normalize_bool(standardize, name="standardize"),
            class_balance_source=_normalize_bool(class_balance_source, name="class_balance_source"),
            normalize_latent=_normalize_bool(normalize_latent, name="normalize_latent"),
        )

    setattr(subspace_adaptation_config, _PATCH_MARKER, True)
    subspace_adaptation.subspace_adaptation_config = subspace_adaptation_config


__all__ = ["install"]
