"""Normalize direct Source MixStyle config construction and coercion."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import replace
from functools import wraps
from typing import Any

import numpy as np

_INSTALLED = False
_PATCH_MARKER = "_neureptrace_source_mixstyle_direct_config_patch_installed"
_INIT_PATCH_MARKER = "_neureptrace_source_mixstyle_direct_config_init_patch_installed"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}
_NONE_STRINGS = {"", "none", "null"}


def _scalar_config_value(value: Any, *, message: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        return value.item()
    if isinstance(value, (list, tuple, dict, set)):
        raise ValueError(message)
    return value


def _normalize_bool(value: Any, *, name: str) -> bool:
    message = f"{name} must be a boolean value."
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        return _normalize_bool(value.item(), name=name)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
        raise ValueError(message)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value) in {0.0, 1.0}:
        return bool(value)
    raise ValueError(message)


def _normalize_nonnegative_int(source_mixstyle: Any, value: Any, *, name: str) -> int:
    message = f"{name} must be a non-negative integer."
    value = _scalar_config_value(value, message=message)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    return source_mixstyle._normalize_nonnegative_int(value, name=name)


def _normalize_positive_float(source_mixstyle: Any, value: Any, *, name: str) -> float:
    message = f"{name} must be a positive finite value."
    value = _scalar_config_value(value, message=message)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    return source_mixstyle._normalize_positive_float(value, name=name)


def _normalize_nonnegative_float(source_mixstyle: Any, value: Any, *, name: str) -> float:
    message = f"{name} must be finite and non-negative."
    value = _scalar_config_value(value, message=message)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    return source_mixstyle._normalize_nonnegative_float(value, name=name)


def _normalize_unit_interval(source_mixstyle: Any, value: Any, *, name: str) -> float:
    message = f"{name} must be in [0, 1]."
    value = _scalar_config_value(value, message=message)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    return source_mixstyle._normalize_unit_interval(value, name=name)


def _is_none_random_state(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _NONE_STRINGS
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _is_none_random_state(value.item())
    return False


def _normalize_optional_random_state(source_mixstyle: Any, value: Any) -> int | None:
    if _is_none_random_state(value):
        return None
    return _normalize_nonnegative_int(source_mixstyle, value, name="random_state")


def _normalized_fields(source_mixstyle: Any, config: Any) -> dict[str, Any]:
    return {
        "mixes_per_row": _normalize_nonnegative_int(source_mixstyle, config.mixes_per_row, name="mixes_per_row"),
        "alpha": _normalize_positive_float(source_mixstyle, config.alpha, name="alpha"),
        "style_strength": _normalize_unit_interval(source_mixstyle, config.style_strength, name="style_strength"),
        "synthetic_weight": _normalize_nonnegative_float(source_mixstyle, config.synthetic_weight, name="synthetic_weight"),
        "include_original": _normalize_bool(config.include_original, name="include_original"),
        "random_state": _normalize_optional_random_state(source_mixstyle, config.random_state),
    }


def _normalize_config(source_mixstyle: Any, config: Any):
    return replace(config, **_normalized_fields(source_mixstyle, config))


def _patch_config_init(source_mixstyle: Any) -> None:
    original_init = source_mixstyle.SourceMixStyleConfig.__init__
    if getattr(original_init, _INIT_PATCH_MARKER, False):
        return

    @wraps(original_init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        for name, value in _normalized_fields(source_mixstyle, self).items():
            object.__setattr__(self, name, value)

    setattr(__init__, _INIT_PATCH_MARKER, True)
    source_mixstyle.SourceMixStyleConfig.__init__ = __init__


def install() -> None:
    """Install Source MixStyle direct-config normalization."""

    global _INSTALLED
    if _INSTALLED:
        return
    source_mixstyle = importlib.import_module("neureptrace.decoding.source_mixstyle")
    _patch_config_init(source_mixstyle)

    original_source_mixstyle_config = source_mixstyle.source_mixstyle_config
    if not getattr(original_source_mixstyle_config, _PATCH_MARKER, False):

        @wraps(original_source_mixstyle_config)
        def source_mixstyle_config(
            *,
            mixes_per_row: int | str = source_mixstyle.DEFAULT_MIXSTYLE_MIXES_PER_ROW,
            alpha: float | str = source_mixstyle.DEFAULT_MIXSTYLE_ALPHA,
            style_strength: float | str = source_mixstyle.DEFAULT_MIXSTYLE_STYLE_STRENGTH,
            synthetic_weight: float | str = source_mixstyle.DEFAULT_MIXSTYLE_SYNTHETIC_WEIGHT,
            include_original: bool = True,
            random_state: int | str | None = 13,
        ):
            config = source_mixstyle.SourceMixStyleConfig(
                mixes_per_row=mixes_per_row,
                alpha=alpha,
                style_strength=style_strength,
                synthetic_weight=synthetic_weight,
                include_original=include_original,
                random_state=random_state,
            )
            return _normalize_config(source_mixstyle, config)

        setattr(source_mixstyle_config, _PATCH_MARKER, True)
        source_mixstyle.source_mixstyle_config = source_mixstyle_config

    original_coerce_config = source_mixstyle._coerce_config
    if not getattr(original_coerce_config, _PATCH_MARKER, False):

        @wraps(original_coerce_config)
        def _coerce_config(config: Any):
            if config is None:
                return source_mixstyle.source_mixstyle_config()
            if isinstance(config, Mapping):
                return source_mixstyle.source_mixstyle_config(**dict(config))
            if isinstance(config, source_mixstyle.SourceMixStyleConfig):
                return _normalize_config(source_mixstyle, config)
            return _normalize_config(source_mixstyle, original_coerce_config(config))

        setattr(_coerce_config, _PATCH_MARKER, True)
        source_mixstyle._coerce_config = _coerce_config

    _INSTALLED = True
