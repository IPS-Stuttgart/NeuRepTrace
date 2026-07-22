"""Keep onset-sensitivity identifiers injective and boolean controls strict."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_SETTING_ID_PATCH_MARKER = "_neureptrace_onset_sensitivity_setting_id_patch_installed"
_BOOLEAN_CONTROL_PATCH_MARKER = "_neureptrace_onset_sensitivity_boolean_control_patch_installed"


def _exact_float_token(value: float) -> str:
    """Return a path-safe token that uniquely identifies one float64 value."""

    return float(value).hex().replace(".", "d").replace("+", "p").replace("-", "m")


def _rounded_float_token(
    value: float,
    *,
    prefix: str,
    scale: int,
    width: int,
    suffix: str = "",
) -> str:
    """Keep compact legacy tokens, disambiguating only lossy rounding."""

    rounded = int(round(value * scale))
    token = f"{prefix}{rounded:0{width}d}{suffix}"
    if value == rounded / scale:
        return token
    return f"{token}x{_exact_float_token(value)}"


def _setting_id(setting: Any) -> str:
    """Return a deterministic path-safe identifier for one sensitivity setting."""

    method = str(setting.threshold_method).replace("_", "")
    quantile = _rounded_float_token(
        float(setting.threshold_quantile),
        prefix="q",
        scale=1000,
        width=4,
    )
    consecutive = f"c{int(setting.min_consecutive):02d}"
    duration = (
        "dnone"
        if setting.min_duration is None
        else _rounded_float_token(
            float(setting.min_duration),
            prefix="d",
            scale=1000,
            width=4,
            suffix="ms",
        )
    )
    stable = "stable" if setting.require_stable_prediction else "anypred"
    return "_".join([method, quantile, consecutive, duration, stable])


def _boolean_value(value: object, *, name: str) -> bool:
    """Return a strict Python boolean without accepting truthy substitutes."""

    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean value.")
    return bool(value)


def _boolean_grid_values(values: object) -> tuple[bool, ...]:
    """Validate stable-prediction grid values before ``bool`` can coerce them."""

    try:
        raw_values = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("stable_prediction_values must contain only boolean values.") from exc
    if any(not isinstance(value, (bool, np.bool_)) for value in raw_values):
        raise ValueError("stable_prediction_values must contain only boolean values.")
    return tuple(bool(value) for value in raw_values)


def _install_boolean_control_validation(onset_sensitivity: Any) -> None:
    """Reject truthy non-booleans in both onset-sensitivity public APIs."""

    original_build_settings = onset_sensitivity.build_sensitivity_settings
    if not getattr(original_build_settings, _BOOLEAN_CONTROL_PATCH_MARKER, False):

        @wraps(original_build_settings)
        def build_sensitivity_settings(*args: Any, **kwargs: Any):
            if "stable_prediction_values" in kwargs:
                kwargs["stable_prediction_values"] = _boolean_grid_values(kwargs["stable_prediction_values"])
            return original_build_settings(*args, **kwargs)

        setattr(build_sensitivity_settings, _BOOLEAN_CONTROL_PATCH_MARKER, True)
        onset_sensitivity.build_sensitivity_settings = build_sensitivity_settings

    original_run_sensitivity = onset_sensitivity.run_onset_sensitivity
    if not getattr(original_run_sensitivity, _BOOLEAN_CONTROL_PATCH_MARKER, False):

        @wraps(original_run_sensitivity)
        def run_onset_sensitivity(*args: Any, **kwargs: Any):
            if "include_stable_prediction" in kwargs:
                kwargs["include_stable_prediction"] = _boolean_value(
                    kwargs["include_stable_prediction"],
                    name="include_stable_prediction",
                )
            return original_run_sensitivity(*args, **kwargs)

        setattr(run_onset_sensitivity, _BOOLEAN_CONTROL_PATCH_MARKER, True)
        onset_sensitivity.run_onset_sensitivity = run_onset_sensitivity


def install() -> None:
    """Install collision-free identifiers and strict boolean sensitivity controls."""

    onset_sensitivity = importlib.import_module("neureptrace.onset_sensitivity")
    current_property = onset_sensitivity.OnsetSensitivitySetting.setting_id
    current_getter = getattr(current_property, "fget", None)
    if not getattr(current_getter, _SETTING_ID_PATCH_MARKER, False):
        setattr(_setting_id, _SETTING_ID_PATCH_MARKER, True)
        onset_sensitivity.OnsetSensitivitySetting.setting_id = property(_setting_id)

    _install_boolean_control_validation(onset_sensitivity)


__all__ = ["install"]
