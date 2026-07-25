"""Normalize generative augmentation config values, feature matrices, and covariance powers."""

from __future__ import annotations

import importlib
from decimal import Decimal, InvalidOperation
from functools import wraps
from numbers import Integral, Real
from typing import Any

import numpy as np

_INSTALLED = False
_CONFIG_PATCH_MARKER = "_neureptrace_generative_augmentation_random_state_patch_installed"
_CONFIG_COERCER_PATCH_MARKER = "_neureptrace_generative_augmentation_direct_config_patch_installed"
_FEATURE_PATCH_MARKER = "_neureptrace_generative_augmentation_finite_feature_patch_installed"
_INTEGER_PATCH_MARKER = "_neureptrace_generative_augmentation_exact_integer_patch_installed"
_MATRIX_POWER_PATCH_MARKER = "_neureptrace_generative_augmentation_zero_floor_matrix_power_patch_installed"


def _random_state_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a non-negative integer or None.")


def _integer_error(name: str) -> ValueError:
    return ValueError(f"{name} must be an integer.")


def _is_none_random_state(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _scalar_random_state_value(value: Any, *, name: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _random_state_error(name)
        return value.item()
    if isinstance(value, (list, tuple, dict, set)):
        raise _random_state_error(name)
    return value


def _decimal_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bytes):
        try:
            text = value.decode().strip()
        except UnicodeDecodeError as exc:
            raise _integer_error(name) from exc
    else:
        text = str(value).strip()
    if not text:
        raise _integer_error(name)
    try:
        numeric = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise _integer_error(name) from exc
    if not numeric.is_finite():
        raise _integer_error(name)
    integral = numeric.to_integral_value()
    if numeric != integral:
        raise _integer_error(name)
    return int(integral)


def _normalize_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise _integer_error(name)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise _integer_error(name)
        integral = value.to_integral_value()
        if value != integral:
            raise _integer_error(name)
        return int(integral)
    if isinstance(value, Real):
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise _integer_error(name)
        return int(numeric)
    return _decimal_integer(value, name=name)


def _normalize_optional_nonnegative_int(value: Any, *, name: str = "random_state") -> int | None:
    value = _scalar_random_state_value(value, name=name)
    if _is_none_random_state(value):
        return None
    try:
        integer = _normalize_integer(value, name=name)
    except ValueError as exc:
        raise _random_state_error(name) from exc
    if integer < 0:
        raise _random_state_error(name)
    return integer


def install() -> None:
    """Patch generative augmentation config and numerical input normalization."""

    global _INSTALLED
    if _INSTALLED:
        return

    module = importlib.import_module("neureptrace.decoding.generative_augmentation")
    if not getattr(module._normalize_integer, _INTEGER_PATCH_MARKER, False):
        setattr(_normalize_integer, _INTEGER_PATCH_MARKER, True)
        module._normalize_integer = _normalize_integer

    original_config = module.generative_augmentation_config
    if not getattr(original_config, _CONFIG_PATCH_MARKER, False):

        @wraps(original_config)
        def generative_augmentation_config(*args: Any, **kwargs: Any):
            if "random_state" in kwargs:
                kwargs = dict(kwargs)
                kwargs["random_state"] = _normalize_optional_nonnegative_int(kwargs["random_state"], name="random_state")
            return original_config(*args, **kwargs)

        setattr(generative_augmentation_config, _CONFIG_PATCH_MARKER, True)
        module.generative_augmentation_config = generative_augmentation_config

    original_coerce_config = module._coerce_config
    if not getattr(original_coerce_config, _CONFIG_COERCER_PATCH_MARKER, False):

        @wraps(original_coerce_config)
        def _coerce_config(config: Any):
            if isinstance(config, module.GenerativeAugmentationConfig):
                return module.generative_augmentation_config(
                    method=config.method,
                    synthetic_per_class=config.synthetic_per_class,
                    noise_scale=config.noise_scale,
                    covariance_shrinkage=config.covariance_shrinkage,
                    covariance_floor=config.covariance_floor,
                    random_state=config.random_state,
                    target_style_strength=config.target_style_strength,
                    target_calibration_weight=config.target_calibration_weight,
                    neural_epochs=config.neural_epochs,
                    neural_hidden_dim=config.neural_hidden_dim,
                    neural_batch_size=config.neural_batch_size,
                    neural_learning_rate=config.neural_learning_rate,
                    gan_latent_dim=config.gan_latent_dim,
                    gan_discriminator_steps=config.gan_discriminator_steps,
                    diffusion_steps=config.diffusion_steps,
                )
            return original_coerce_config(config)

        setattr(_coerce_config, _CONFIG_COERCER_PATCH_MARKER, True)
        module._coerce_config = _coerce_config

    original_feature_matrix = module._feature_matrix
    if not getattr(original_feature_matrix, _FEATURE_PATCH_MARKER, False):

        @wraps(original_feature_matrix)
        def _feature_matrix(features: Any, *, name: str):
            matrix = original_feature_matrix(features, name=name)
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} must contain only finite values.")
            return matrix

        setattr(_feature_matrix, _FEATURE_PATCH_MARKER, True)
        module._feature_matrix = _feature_matrix

    original_matrix_power = module._matrix_power
    if not getattr(original_matrix_power, _MATRIX_POWER_PATCH_MARKER, False):

        @wraps(original_matrix_power)
        def _matrix_power(matrix: np.ndarray, power: float, *, floor: float) -> np.ndarray:
            numeric_power = float(power)
            numeric_floor = max(float(floor), 0.0)
            if numeric_power >= 0.0 or numeric_floor > 0.0:
                return original_matrix_power(matrix, numeric_power, floor=numeric_floor)

            array = np.asarray(matrix, dtype=float)
            symmetric = 0.5 * (array + array.T)
            values, vectors = np.linalg.eigh(symmetric)
            scale = float(np.max(np.abs(values))) if values.size else 0.0
            tolerance = np.finfo(float).eps * max(symmetric.shape) * scale
            powered_values = np.zeros_like(values)
            positive = values > tolerance
            powered_values[positive] = np.power(values[positive], numeric_power)
            return (vectors * powered_values) @ vectors.T

        setattr(_matrix_power, _MATRIX_POWER_PATCH_MARKER, True)
        module._matrix_power = _matrix_power

    _INSTALLED = True


__all__ = ["install"]
