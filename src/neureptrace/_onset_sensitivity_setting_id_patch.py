"""Keep onset-sensitivity setting identifiers injective."""

from __future__ import annotations

import importlib
from typing import Any

_PATCH_MARKER = "_neureptrace_onset_sensitivity_setting_id_patch_installed"


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


def install() -> None:
    """Install collision-free setting identifiers for onset sensitivity sweeps."""

    onset_sensitivity = importlib.import_module("neureptrace.onset_sensitivity")
    current_property = onset_sensitivity.OnsetSensitivitySetting.setting_id
    current_getter = getattr(current_property, "fget", None)
    if getattr(current_getter, _PATCH_MARKER, False):
        return

    setattr(_setting_id, _PATCH_MARKER, True)
    onset_sensitivity.OnsetSensitivitySetting.setting_id = property(_setting_id)


__all__ = ["install"]
