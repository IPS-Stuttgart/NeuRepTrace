from __future__ import annotations

from typing import Any

import numpy as np

from neureptrace.decoding import adaptive_normalization as _adaptive_normalization
from neureptrace.decoding import calibrated_prototypes as _calibrated_prototypes

_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}


def _bool_config(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
        raise ValueError(f"{name} must be a boolean value.")
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a boolean value.")
        return _bool_config(value.item(), name=name)
    if isinstance(value, (int, np.integer)):
        if int(value) in {0, 1}:
            return bool(value)
        raise ValueError(f"{name} must be a boolean value.")
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value) in {0.0, 1.0}:
            return bool(value)
        raise ValueError(f"{name} must be a boolean value.")
    raise ValueError(f"{name} must be a boolean value.")


def _patched_adaptive_normalization_config(
    *,
    mode: str | None = "domain_wise",
    center: object = True,
    scale: object = True,
    robust: object = False,
    epsilon: float | str = _adaptive_normalization.DEFAULT_EPSILON,
) -> _adaptive_normalization.AdaptiveNormalizationConfig:
    return _adaptive_normalization.AdaptiveNormalizationConfig(
        mode=_adaptive_normalization.normalize_adaptive_normalization_mode(mode),
        center=_bool_config(center, name="center"),
        scale=_bool_config(scale, name="scale"),
        robust=_bool_config(robust, name="robust"),
        epsilon=_adaptive_normalization._positive_float(epsilon, name="epsilon"),
    )


def _patched_calibrated_prototype_config(
    *,
    prior_strength: float | str = _calibrated_prototypes.DEFAULT_PRIOR_STRENGTH,
    fixed_calibration_weight: float | str | None = None,
    temperature: float | str = _calibrated_prototypes.DEFAULT_TEMPERATURE,
    diagonal_scale: object = True,
    epsilon: float | str = _calibrated_prototypes.DEFAULT_EPSILON,
) -> _calibrated_prototypes.CalibratedPrototypeConfig:
    fixed = None if fixed_calibration_weight in {None, "", "none", "None"} else _calibrated_prototypes._unit_interval_float(fixed_calibration_weight, name="fixed_calibration_weight")
    return _calibrated_prototypes.CalibratedPrototypeConfig(
        prior_strength=_calibrated_prototypes._positive_float(prior_strength, name="prior_strength"),
        fixed_calibration_weight=fixed,
        temperature=_calibrated_prototypes._positive_float(temperature, name="temperature"),
        diagonal_scale=_bool_config(diagonal_scale, name="diagonal_scale"),
        epsilon=_calibrated_prototypes._positive_float(epsilon, name="epsilon"),
    )


def install() -> None:
    _adaptive_normalization.adaptive_normalization_config = _patched_adaptive_normalization_config
    _calibrated_prototypes.calibrated_prototype_config = _patched_calibrated_prototype_config
