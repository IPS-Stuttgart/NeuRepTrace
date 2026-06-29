"""Reject malformed Source Bagging numeric options."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_bagging_fraction_bounds_patch_installed"


def _fraction_error(name: str) -> ValueError:
    return ValueError(f"{name} must be in (0, 1].")


def _positive_float_error(name: str) -> ValueError:
    return ValueError(f"{name} must be positive and finite.")


def _bounded_fraction(value: Any, *, name: str) -> float:
    """Return a finite fraction in ``(0, 1]`` while rejecting booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise _fraction_error(name)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _fraction_error(name)
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise _fraction_error(name)
    if isinstance(value, (list, tuple, dict, set)):
        raise _fraction_error(name)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise _fraction_error(name) from exc
    if not np.isfinite(parsed) or parsed <= 0.0 or parsed > 1.0:
        raise _fraction_error(name)
    return parsed


def _positive_float(value: Any, *, name: str) -> float:
    """Return a positive finite scalar while rejecting boolean and array controls."""

    if isinstance(value, (bool, np.bool_)):
        raise _positive_float_error(name)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _positive_float_error(name)
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise _positive_float_error(name)
    if isinstance(value, (list, tuple, dict, set)):
        raise _positive_float_error(name)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise _positive_float_error(name) from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise _positive_float_error(name)
    return parsed


def _validate_config(cfg: Any) -> Any:
    _bounded_fraction(cfg.sample_fraction, name="sample_fraction")
    _bounded_fraction(cfg.feature_fraction, name="feature_fraction")
    _positive_float(cfg.epsilon, name="epsilon")
    return cfg


def install() -> None:
    """Install source-bagging numeric-option validation."""

    source_bagging = importlib.import_module("neureptrace.decoding.source_bagging")

    original_config = source_bagging.source_bagging_config
    if not getattr(original_config, _PATCH_MARKER, False):

        @wraps(original_config)
        def source_bagging_config(
            *,
            n_estimators: Any = source_bagging.DEFAULT_N_ESTIMATORS,
            sample_fraction: Any = source_bagging.DEFAULT_SAMPLE_FRACTION,
            feature_fraction: Any = source_bagging.DEFAULT_FEATURE_FRACTION,
            bootstrap_rows: Any = True,
            bootstrap_features: Any = False,
            class_balanced: Any = True,
            random_state: Any = 13,
            epsilon: Any = source_bagging.DEFAULT_EPSILON,
        ):
            return _validate_config(
                original_config(
                    n_estimators=n_estimators,
                    sample_fraction=_bounded_fraction(sample_fraction, name="sample_fraction"),
                    feature_fraction=_bounded_fraction(feature_fraction, name="feature_fraction"),
                    bootstrap_rows=bootstrap_rows,
                    bootstrap_features=bootstrap_features,
                    class_balanced=class_balanced,
                    random_state=random_state,
                    epsilon=_positive_float(epsilon, name="epsilon"),
                )
            )

        setattr(source_bagging_config, _PATCH_MARKER, True)
        source_bagging.source_bagging_config = source_bagging_config

    original_coerce_config = source_bagging._coerce_config
    if not getattr(original_coerce_config, _PATCH_MARKER, False):

        @wraps(original_coerce_config)
        def _coerce_config(config: Any):
            return _validate_config(original_coerce_config(config))

        setattr(_coerce_config, _PATCH_MARKER, True)
        source_bagging._coerce_config = _coerce_config


install()

__all__ = ["install"]
