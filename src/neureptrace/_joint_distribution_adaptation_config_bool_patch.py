"""Normalize Joint Distribution Adaptation configuration and class inputs."""

from __future__ import annotations

import importlib
import math
from functools import wraps
from numbers import Real
from typing import Any

_PATCH_MARKER = "_neureptrace_jda_boolean_config_patch_installed"
_CLASS_SET_PATCH_MARKER = "_neureptrace_jda_exact_class_set_patch_installed"
_TRUE_STRINGS = {"1", "true", "yes", "on"}
_FALSE_STRINGS = {"0", "false", "no", "off"}


def install() -> None:
    """Install JDA configuration normalization and source-class validation."""

    jda = importlib.import_module("neureptrace.decoding.joint_distribution_adaptation")

    original_config = jda.joint_distribution_adaptation_config
    if not getattr(original_config, _PATCH_MARKER, False):

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
            standardize: bool | int | float | str = True,
            normalize_latent: bool | int | float | str = False,
        ) -> Any:
            _reject_boolean_numeric(n_components, name="n_components", expected="positive integer")
            _reject_boolean_numeric(max_iterations, name="max_iterations", expected="positive integer")
            _reject_boolean_numeric(conditional_weight, name="conditional_weight", expected="finite and non-negative")
            _reject_boolean_numeric(regularization, name="regularization", expected="finite and non-negative")
            _reject_boolean_numeric(eigen_ridge, name="eigen_ridge", expected="positive and finite")
            _reject_boolean_numeric(temperature, name="temperature", expected="positive and finite")
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

    original_resolve_classes = getattr(jda, "_resolve_classes")
    if not getattr(original_resolve_classes, _CLASS_SET_PATCH_MARKER, False):

        @wraps(original_resolve_classes)
        def _resolve_classes(labels: Any, classes: Any) -> tuple[Any, ...]:
            resolved = original_resolve_classes(labels, classes)
            if classes is None:
                return resolved

            inferred = tuple(dict.fromkeys(labels.tolist()))
            extras = [value for value in resolved if value not in inferred]
            if extras:
                raise ValueError(
                    "classes contain label(s) absent from source_labels: "
                    f"{extras!r}. JDA requires at least one source row for every supplied class."
                )
            return resolved

        setattr(_resolve_classes, _CLASS_SET_PATCH_MARKER, True)
        setattr(jda, "_resolve_classes", _resolve_classes)


def _normalize_bool(value: bool | int | float | str, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Real):
        numeric = float(value)
        if math.isfinite(numeric) and numeric in {0.0, 1.0}:
            return bool(numeric)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    raise ValueError(f"{name} must be a boolean.")


def _reject_boolean_numeric(value: Any, *, name: str, expected: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be {expected}.")


__all__ = ["install"]
