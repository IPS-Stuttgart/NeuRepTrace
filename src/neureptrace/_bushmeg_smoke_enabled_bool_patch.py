"""Normalize BUSH-MEG all-protocol method-setting booleans."""

from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_bushmeg_smoke_enabled_bool_patch_installed"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}
_BOOLEAN_METHOD_SETTINGS = {
    "enabled": True,
    "heavy": False,
    "smoke_enabled": False,
}


def _bool_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a boolean value.")


def _normalize_bool(value: Any, *, name: str, default: bool = False) -> bool:
    """Return a real bool for user-facing YAML/CLI-style config values."""

    if value is None:
        return bool(default)
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
        return _normalize_bool(value.item(), name=name, default=default)
    if isinstance(value, (int, np.integer)):
        if int(value) in {0, 1}:
            return bool(value)
        raise _bool_error(name)
    raise _bool_error(name)


def _normalize_method_setting_bools(settings: Mapping[str, Any], *, method: str) -> dict[str, Any]:
    """Normalize boolean-valued method settings while preserving all others."""

    normalized = dict(settings)
    for key, default in _BOOLEAN_METHOD_SETTINGS.items():
        if key in normalized:
            normalized[key] = _normalize_bool(
                normalized[key],
                name=f"all_protocols.method_settings.{method}.{key}",
                default=default,
            )
    return normalized


def _with_normalized_method_settings(all_protocols_config: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    settings_block = all_protocols_config.get("method_settings", {})
    if not isinstance(settings_block, Mapping):
        return all_protocols_config
    settings = settings_block.get(method, {})
    if not isinstance(settings, Mapping) or not any(key in settings for key in _BOOLEAN_METHOD_SETTINGS):
        return all_protocols_config

    normalized_config = copy.deepcopy(dict(all_protocols_config))
    normalized_settings_block = copy.deepcopy(dict(settings_block))
    normalized_settings_block[method] = _normalize_method_setting_bools(settings, method=method)
    normalized_config["method_settings"] = normalized_settings_block
    return normalized_config


def _with_normalized_smoke_enabled(all_protocols_config: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    """Backward-compatible alias for the original smoke-enabled-only helper."""

    return _with_normalized_method_settings(all_protocols_config, method)


def install() -> None:
    """Patch all-protocol method settings so quoted booleans keep boolean meaning."""

    all_protocols = importlib.import_module("neureptrace.bushmeg_all_protocols")

    original_method_settings = all_protocols._method_settings
    if not getattr(original_method_settings, _PATCH_MARKER, False):

        @wraps(original_method_settings)
        def _method_settings(all_protocols_config: Mapping[str, Any], method: str) -> dict[str, Any]:
            normalized_config = all_protocols_config
            if isinstance(all_protocols_config, Mapping):
                normalized_config = _with_normalized_method_settings(all_protocols_config, method)
            return original_method_settings(normalized_config, method)

        setattr(_method_settings, _PATCH_MARKER, True)
        all_protocols._method_settings = _method_settings

    original_method_config = all_protocols._method_config
    if not getattr(original_method_config, _PATCH_MARKER, False):

        @wraps(original_method_config)
        def _method_config(
            base_config: Mapping[str, Any],
            all_protocols_config: Mapping[str, Any],
            spec: Any,
            *args: Any,
            **kwargs: Any,
        ):
            normalized_config = all_protocols_config
            if isinstance(all_protocols_config, Mapping):
                normalized_config = _with_normalized_method_settings(all_protocols_config, spec.method)
            return original_method_config(base_config, normalized_config, spec, *args, **kwargs)

        setattr(_method_config, _PATCH_MARKER, True)
        all_protocols._method_config = _method_config

    setattr(all_protocols, _PATCH_MARKER, True)


__all__ = ["install"]
