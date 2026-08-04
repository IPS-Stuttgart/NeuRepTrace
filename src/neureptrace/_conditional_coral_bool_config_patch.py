"""Normalize conditional-CORAL config and reject lossy complex inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

import numpy as np

_CONFIG_PATCH_MARKER = "_neureptrace_conditional_coral_bool_config_patch_installed"
_FEATURE_PATCH_MARKER = "_neureptrace_conditional_coral_complex_feature_patch_installed"
_PROBABILITY_PATCH_MARKER = "_neureptrace_conditional_coral_complex_probability_patch_installed"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}
_NONE_STRINGS = {"", "none", "null"}


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


def _random_state_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a non-negative integer or None.")


def _normalize_optional_random_state(value: Any, *, name: str) -> int | None:
    """Normalize optional integer seeds without leaking raw set/NumPy errors."""

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in _NONE_STRINGS:
            return None
        value = stripped
    elif isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _random_state_error(name)
        return _normalize_optional_random_state(value.item(), name=name)
    elif isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
        raise _random_state_error(name)
    if isinstance(value, (bool, np.bool_)):
        raise _random_state_error(name)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise _random_state_error(name) from exc
    if not np.isfinite(parsed) or parsed < 0.0 or parsed % 1.0 != 0.0:
        raise _random_state_error(name)
    return int(parsed)


def _contains_complex_value(value: Any) -> bool:
    """Return whether a declared numeric container contains complex values."""

    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return bool(value.size)
        if value.dtype == object:
            return any(_contains_complex_value(item) for item in value.ravel(order="C"))
        return False
    if isinstance(value, np.generic):
        return _contains_complex_value(value.item())
    if hasattr(value, "__array__"):
        try:
            return _contains_complex_value(np.asarray(value))
        except (TypeError, ValueError):
            return False
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return False
    if isinstance(value, Sequence):
        return any(_contains_complex_value(item) for item in value)
    return False


def install() -> None:
    """Install strict config and real-valued input validation for Conditional CORAL."""

    from neureptrace.decoding import conditional_coral

    original_config = conditional_coral.conditional_coral_config
    if not getattr(original_config, _CONFIG_PATCH_MARKER, False):

        @wraps(original_config)
        def conditional_coral_config(
            *,
            regularization: float | str = conditional_coral.DEFAULT_CONDITIONAL_CORAL_REGULARIZATION,
            min_target_rows_per_class: int | str = conditional_coral.DEFAULT_CONDITIONAL_CORAL_MIN_TARGET_ROWS,
            confidence_threshold: float | str = 0.0,
            fallback: str = "global",
            center: Any = True,
            random_state: Any = 13,
        ):
            return original_config(
                regularization=regularization,
                min_target_rows_per_class=min_target_rows_per_class,
                confidence_threshold=confidence_threshold,
                fallback=fallback,
                center=_normalize_bool(center, name="center"),
                random_state=_normalize_optional_random_state(random_state, name="random_state"),
            )

        setattr(conditional_coral_config, _CONFIG_PATCH_MARKER, True)
        conditional_coral.conditional_coral_config = conditional_coral_config

    original_feature_matrix = conditional_coral._feature_matrix
    if not getattr(original_feature_matrix, _FEATURE_PATCH_MARKER, False):

        @wraps(original_feature_matrix)
        def _feature_matrix(values: Any, *, name: str) -> np.ndarray:
            if _contains_complex_value(values):
                raise ValueError(
                    f"{name} must contain real-valued feature values, not complex values."
                )
            return original_feature_matrix(values, name=name)

        setattr(_feature_matrix, _FEATURE_PATCH_MARKER, True)
        conditional_coral._feature_matrix = _feature_matrix

    original_fit = conditional_coral.fit_pseudo_label_conditional_coral
    if not getattr(original_fit, _PROBABILITY_PATCH_MARKER, False):

        @wraps(original_fit)
        def fit_pseudo_label_conditional_coral(
            *,
            source_features: Any,
            source_labels: Any,
            target_features: Any,
            config: Any = None,
            estimator: Any = None,
            target_pseudo_labels: Any = None,
            target_probabilities: Any = None,
        ):
            if target_probabilities is not None and _contains_complex_value(target_probabilities):
                raise ValueError(
                    "target_probabilities must contain real-valued probability values, not complex values."
                )
            return original_fit(
                source_features=source_features,
                source_labels=source_labels,
                target_features=target_features,
                config=config,
                estimator=estimator,
                target_pseudo_labels=target_pseudo_labels,
                target_probabilities=target_probabilities,
            )

        setattr(fit_pseudo_label_conditional_coral, _PROBABILITY_PATCH_MARKER, True)
        conditional_coral.fit_pseudo_label_conditional_coral = fit_pseudo_label_conditional_coral


__all__ = ["install"]
