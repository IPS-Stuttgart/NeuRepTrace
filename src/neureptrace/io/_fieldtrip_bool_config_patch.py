"""Normalize generic FieldTrip loader boolean configuration values.

The public FieldTrip config path is commonly populated from YAML/JSON files.  A
quoted value such as ``"false"`` must therefore keep its boolean meaning instead
of becoming truthy through Python's ``bool("false")`` coercion.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Mapping
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_fieldtrip_bool_config_patch_installed"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}
_BOOL_FIELDS = (
    "trim_channel_labels_to_data",
    "require_equal_trial_time_lengths",
    "require_trialinfo_rows_equal_trials",
)


def _bool_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a boolean value.")


def _normalize_bool(value: Any, *, name: str) -> bool:
    """Return a real bool for user-facing YAML/JSON-style config values."""

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
        parsed = int(value)
        if parsed in {0, 1}:
            return bool(parsed)
        raise _bool_error(name)
    raise _bool_error(name)


def _normalize_validation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize FieldTrip validation booleans before the legacy bool() calls."""

    normalized_config = dict(config)
    validation = normalized_config.get("validation", {}) or {}
    if not isinstance(validation, Mapping):
        return normalized_config

    normalized_validation = dict(validation)
    for key in _BOOL_FIELDS:
        if key in normalized_validation:
            normalized_validation[key] = _normalize_bool(normalized_validation[key], name=f"validation.{key}")

    # ``trim_channel_labels_to_data`` also has a historical top-level alias.  Move
    # a normalized value into ``validation`` so the original implementation's
    # ``validation.get(..., config.get(...))`` expression never sees a quoted
    # string and accidentally treats it as truthy.
    if "trim_channel_labels_to_data" not in normalized_validation and "trim_channel_labels_to_data" in normalized_config:
        normalized_validation["trim_channel_labels_to_data"] = _normalize_bool(
            normalized_config["trim_channel_labels_to_data"],
            name="trim_channel_labels_to_data",
        )

    normalized_config["validation"] = normalized_validation
    return normalized_config


def _normalized_spec_args(original_spec: type[Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    field_names = tuple(field.name for field in dataclasses.fields(original_spec))
    args_list = list(args)
    normalized_kwargs = dict(kwargs)
    for field_name in _BOOL_FIELDS:
        position = field_names.index(field_name)
        if position < len(args_list):
            args_list[position] = _normalize_bool(args_list[position], name=f"FieldTripMatSpec.{field_name}")
        if field_name in normalized_kwargs:
            normalized_kwargs[field_name] = _normalize_bool(normalized_kwargs[field_name], name=f"FieldTripMatSpec.{field_name}")
    return tuple(args_list), normalized_kwargs


def install() -> None:
    """Patch generic FieldTrip config entry points once."""

    fieldtrip_mat = importlib.import_module("neureptrace.io.fieldtrip_mat")
    if getattr(fieldtrip_mat, _PATCH_MARKER, False):
        return

    original_spec = fieldtrip_mat.FieldTripMatSpec

    class FieldTripMatSpec(original_spec):  # type: ignore[misc, valid-type]
        """Compatibility subclass that normalizes boolean constructor aliases."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            normalized_args, normalized_kwargs = _normalized_spec_args(original_spec, args, kwargs)
            super().__init__(*normalized_args, **normalized_kwargs)

    FieldTripMatSpec.__name__ = original_spec.__name__
    FieldTripMatSpec.__qualname__ = original_spec.__qualname__
    FieldTripMatSpec.__module__ = original_spec.__module__
    fieldtrip_mat.FieldTripMatSpec = FieldTripMatSpec

    original_loader = fieldtrip_mat.load_fieldtrip_mat_epochs

    @wraps(original_loader)
    def load_fieldtrip_mat_epochs(path: str | Any, config: Mapping[str, Any] | None = None, *args: Any, **kwargs: Any):
        normalized_config = _normalize_validation_config(config) if isinstance(config, Mapping) else config
        return original_loader(path, normalized_config, *args, **kwargs)

    setattr(load_fieldtrip_mat_epochs, _PATCH_MARKER, True)
    fieldtrip_mat.load_fieldtrip_mat_epochs = load_fieldtrip_mat_epochs
    setattr(fieldtrip_mat, _PATCH_MARKER, True)


__all__ = ["install"]
