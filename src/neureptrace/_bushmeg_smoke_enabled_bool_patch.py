"""Normalize BUSH-MEG all-protocol smoke-enabled config values."""

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


def _with_normalized_smoke_enabled(all_protocols_config: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    settings_block = all_protocols_config.get("method_settings", {})
    if not isinstance(settings_block, Mapping):
        return all_protocols_config
    settings = settings_block.get(method, {})
    if not isinstance(settings, Mapping) or "smoke_enabled" not in settings:
        return all_protocols_config

    normalized_config = copy.deepcopy(dict(all_protocols_config))
    normalized_settings_block = copy.deepcopy(dict(settings_block))
    normalized_settings = dict(settings)
    normalized_settings["smoke_enabled"] = _normalize_bool(
        normalized_settings["smoke_enabled"],
        name=f"all_protocols.method_settings.{method}.smoke_enabled",
    )
    normalized_settings_block[method] = normalized_settings
    normalized_config["method_settings"] = normalized_settings_block
    return normalized_config


def install() -> None:
    """Patch all-protocol method config so quoted false values stay false."""

    all_protocols = importlib.import_module("neureptrace.bushmeg_all_protocols")
    original_method_config = all_protocols._method_config
    if getattr(original_method_config, _PATCH_MARKER, False):
        return

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
            normalized_config = _with_normalized_smoke_enabled(all_protocols_config, spec.method)
        return original_method_config(base_config, normalized_config, spec, *args, **kwargs)

    setattr(_method_config, _PATCH_MARKER, True)
    all_protocols._method_config = _method_config
    setattr(all_protocols, _PATCH_MARKER, True)


__all__ = ["install"]
