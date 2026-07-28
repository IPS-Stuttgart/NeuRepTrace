"""Normalize SourceRFF configuration and reject complex feature inputs."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from functools import wraps
from typing import Any

import numpy as np

_INSTALLED = False
_CONFIG_PATCH_MARKER = "_neureptrace_source_rff_none_random_state_config_patch_installed"
_INIT_PATCH_MARKER = "_neureptrace_source_rff_none_random_state_init_patch_installed"
_FEATURE_MATRIX_PATCH_MARKER = "_neureptrace_source_rff_complex_feature_validation_patch_installed"
_NONE_RANDOM_STATE_TOKENS = {"", "none", "null"}


def _random_state_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a non-negative integer.")


def _is_none_like_random_state(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _NONE_RANDOM_STATE_TOKENS
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return False
        return _is_none_like_random_state(value.item())
    if isinstance(value, np.generic):
        return _is_none_like_random_state(value.item())
    return False


def _scalar_random_state_value(value: Any, *, name: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _random_state_error(name)
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, dict, set)):
        raise _random_state_error(name)
    return value


def _normalize_optional_random_state(source_rff: Any, value: Any, *, name: str = "random_state") -> int | None:
    if _is_none_like_random_state(value):
        return None
    scalar_value = _scalar_random_state_value(value, name=name)
    if _is_none_like_random_state(scalar_value):
        return None
    if isinstance(scalar_value, (bool, np.bool_)):
        raise _random_state_error(name)
    return source_rff._nonnegative_int(scalar_value, name=name)


def _materialize_reusable_feature_input(value: object) -> object:
    """Materialize nested one-pass feature iterables before validation."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        materialized = [_materialize_reusable_feature_input(item) for item in value.ravel(order="C")]
        return np.asarray(materialized, dtype=object).reshape(value.shape)
    if isinstance(value, (str, bytes)):
        return value
    if hasattr(value, "__array__"):
        return value
    if not isinstance(value, Iterable):
        return value
    return [_materialize_reusable_feature_input(item) for item in value]


def _contains_complex_value(value: object) -> bool:
    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return bool(value.size)
        if value.dtype == object:
            return any(_contains_complex_value(item) for item in value.ravel(order="C"))
        return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.generic):
        return isinstance(value.item(), complex)
    if hasattr(value, "__array__"):
        try:
            return _contains_complex_value(np.asarray(value))
        except (TypeError, ValueError):
            return False
    if isinstance(value, Iterable):
        return any(_contains_complex_value(item) for item in value)
    return False


def _patch_feature_matrix(source_rff: Any) -> None:
    original_feature_matrix = source_rff._feature_matrix
    if getattr(original_feature_matrix, _FEATURE_MATRIX_PATCH_MARKER, False):
        return

    @wraps(original_feature_matrix)
    def _feature_matrix(values: object, *, name: str) -> np.ndarray:
        materialized = _materialize_reusable_feature_input(values)
        if _contains_complex_value(materialized):
            raise ValueError(f"{name} must contain real-valued feature values, not complex values.")
        return original_feature_matrix(materialized, name=name)

    setattr(_feature_matrix, _FEATURE_MATRIX_PATCH_MARKER, True)
    source_rff._feature_matrix = _feature_matrix


def _patch_config_init(source_rff: Any) -> None:
    original_init = source_rff.SourceRFFConfig.__init__
    if getattr(original_init, _INIT_PATCH_MARKER, False):
        return

    @wraps(original_init)
    def __init__(
        self: Any,
        n_components: int | str = source_rff.DEFAULT_COMPONENTS,
        gamma: float | str = "scale",
        random_state: int | str | None = source_rff.DEFAULT_RANDOM_STATE,
        standardize: bool | int | str = False,
        epsilon: float | str = source_rff.DEFAULT_EPSILON,
    ) -> None:
        object.__setattr__(self, "n_components", source_rff._positive_int(n_components, name="n_components"))
        object.__setattr__(self, "gamma", source_rff.normalize_gamma(gamma))
        object.__setattr__(self, "random_state", _normalize_optional_random_state(source_rff, random_state))
        object.__setattr__(self, "standardize", source_rff._bool_value(standardize, name="standardize"))
        object.__setattr__(self, "epsilon", source_rff._positive_float(epsilon, name="epsilon"))

    setattr(__init__, _INIT_PATCH_MARKER, True)
    source_rff.SourceRFFConfig.__init__ = __init__


def install() -> None:
    """Install SourceRFF input and random-state normalization."""

    global _INSTALLED
    if _INSTALLED:
        return

    source_rff = importlib.import_module("neureptrace.decoding.source_rff")
    _patch_feature_matrix(source_rff)
    _patch_config_init(source_rff)

    original_config = source_rff.source_rff_config
    if not getattr(original_config, _CONFIG_PATCH_MARKER, False):

        @wraps(original_config)
        def source_rff_config(
            *,
            n_components: int | str = source_rff.DEFAULT_COMPONENTS,
            gamma: float | str = "scale",
            random_state: int | str | None = source_rff.DEFAULT_RANDOM_STATE,
            standardize: bool | int | str = False,
            epsilon: float | str = source_rff.DEFAULT_EPSILON,
        ):
            return source_rff.SourceRFFConfig(
                n_components=n_components,
                gamma=gamma,
                random_state=random_state,
                standardize=standardize,
                epsilon=epsilon,
            )

        setattr(source_rff_config, _CONFIG_PATCH_MARKER, True)
        source_rff.source_rff_config = source_rff_config

    _INSTALLED = True


__all__ = ["install"]
