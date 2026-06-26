"""Normalize boolean config values from CLI/YAML-style inputs."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_SUBSPACE_PATCH_MARKER = "_neureptrace_subspace_bool_config_patch_installed"
_CONDITIONAL_CORAL_PATCH_MARKER = "_neureptrace_conditional_coral_bool_config_patch_installed"
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


def _install_subspace_bool_config() -> None:
    from neureptrace.decoding import subspace_adaptation as subspace

    original_config = subspace.subspace_adaptation_config
    if getattr(original_config, _SUBSPACE_PATCH_MARKER, False):
        return

    @wraps(original_config)
    def subspace_adaptation_config(
        *,
        method: str | None = subspace.DEFAULT_SUBSPACE_METHOD,
        n_components: int | str | None = subspace.DEFAULT_SUBSPACE_COMPONENTS,
        regularization: float | str = subspace.DEFAULT_SUBSPACE_REGULARIZATION,
        eigen_ridge: float | str = subspace.DEFAULT_SUBSPACE_EIGEN_RIDGE,
        standardize: Any = True,
        class_balance_source: Any = False,
        normalize_latent: Any = False,
    ):
        normalized_method = subspace.normalize_subspace_method(method)
        requested_balance = _normalize_bool(class_balance_source, name="class_balance_source")
        return original_config(
            method=normalized_method,
            n_components=n_components,
            regularization=regularization,
            eigen_ridge=eigen_ridge,
            standardize=_normalize_bool(standardize, name="standardize"),
            class_balance_source=(requested_balance or normalized_method == "balanced_tca"),
            normalize_latent=_normalize_bool(normalize_latent, name="normalize_latent"),
        )

    setattr(subspace_adaptation_config, _SUBSPACE_PATCH_MARKER, True)
    subspace.subspace_adaptation_config = subspace_adaptation_config


def _install_conditional_coral_bool_config() -> None:
    from neureptrace.decoding import conditional_coral

    original_config = conditional_coral.conditional_coral_config
    if getattr(original_config, _CONDITIONAL_CORAL_PATCH_MARKER, False):
        return

    @wraps(original_config)
    def conditional_coral_config(
        *,
        regularization: float | str = conditional_coral.DEFAULT_CONDITIONAL_CORAL_REGULARIZATION,
        min_target_rows_per_class: int | str = conditional_coral.DEFAULT_CONDITIONAL_CORAL_MIN_TARGET_ROWS,
        confidence_threshold: float | str = 0.0,
        fallback: str = "global",
        center: Any = True,
        random_state: int | str | None = 13,
    ):
        return original_config(
            regularization=regularization,
            min_target_rows_per_class=min_target_rows_per_class,
            confidence_threshold=confidence_threshold,
            fallback=fallback,
            center=_normalize_bool(center, name="center"),
            random_state=random_state,
        )

    setattr(conditional_coral_config, _CONDITIONAL_CORAL_PATCH_MARKER, True)
    conditional_coral.conditional_coral_config = conditional_coral_config


def install() -> None:
    """Install strict boolean normalization for config constructors."""

    _install_subspace_bool_config()
    _install_conditional_coral_bool_config()


__all__ = ["install"]
