"""Runtime patches for source-decoder numeric scalar configuration validation."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any

import numpy as np

_ORIGINAL_COERCE_CONFIG = None
_INSTALLED = False
_SOURCE_KNN_DATACLASS_INIT_PATCH_MARKER = "_neureptrace_source_knn_dataclass_init_patch_installed"
_SOURCE_TEMPERATURE_DATACLASS_INIT_PATCH_MARKER = "_neureptrace_source_temperature_dataclass_init_patch_installed"


def install() -> None:
    """Install source-decoder numeric config validation patches."""
    global _INSTALLED
    if _INSTALLED:
        return

    _install_source_centroid_patch()
    _install_source_knn_patch()
    _install_source_temperature_patch()
    _INSTALLED = True


def _install_source_centroid_patch() -> None:
    """Patch source-centroid config validation and missing-aware label equality."""
    global _ORIGINAL_COERCE_CONFIG

    from neureptrace._object_label_utils import values_equal
    from neureptrace.decoding import source_centroid as module

    _ORIGINAL_COERCE_CONFIG = module._coerce_config
    module.source_centroid_config = _patched_source_centroid_config
    module._coerce_config = _patched_coerce_config
    module._labels_equal = values_equal


def _install_source_knn_patch() -> None:
    """Normalize direct SourceKNNConfig construction like source_knn_config(...)."""

    from neureptrace.decoding import source_knn as module

    original_init = module.SourceKNNConfig.__init__
    if getattr(original_init, _SOURCE_KNN_DATACLASS_INIT_PATCH_MARKER, False):
        return

    @wraps(original_init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        object.__setattr__(self, "k", module._normalize_k_request(self.k))
        object.__setattr__(self, "weights", module.normalize_weight_mode(self.weights))
        object.__setattr__(self, "standardize", module._bool_value(self.standardize, name="standardize"))
        object.__setattr__(self, "epsilon", module._positive_float(self.epsilon, name="epsilon"))

    setattr(__init__, _SOURCE_KNN_DATACLASS_INIT_PATCH_MARKER, True)
    module.SourceKNNConfig.__init__ = __init__


def _install_source_temperature_patch() -> None:
    """Normalize direct SourceTemperatureConfig construction like source_temperature_config(...)."""

    from neureptrace.decoding import source_temperature as module

    original_init = module.SourceTemperatureConfig.__init__
    if getattr(original_init, _SOURCE_TEMPERATURE_DATACLASS_INIT_PATCH_MARKER, False):
        return

    @wraps(original_init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        object.__setattr__(self, "temperatures", _source_temperature_grid(module, self.temperatures))
        object.__setattr__(self, "epsilon", module._probability_epsilon(self.epsilon))

    setattr(__init__, _SOURCE_TEMPERATURE_DATACLASS_INIT_PATCH_MARKER, True)
    module.SourceTemperatureConfig.__init__ = __init__


def _patched_source_centroid_config(
    *,
    temperature: float | str = 1.0,
    use_diagonal_scale: bool | str | int | float = True,
    shrinkage: float | str = 0.0,
    epsilon: float | str = 1e-8,
):
    from neureptrace.decoding import source_centroid as module

    return module.SourceCentroidConfig(
        temperature=_positive_float(temperature, name="temperature"),
        use_diagonal_scale=module._boolean(use_diagonal_scale, name="use_diagonal_scale"),
        shrinkage=_unit_interval_float(shrinkage, name="shrinkage"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def _patched_coerce_config(config: Any):
    from neureptrace.decoding import source_centroid as module

    if isinstance(config, module.SourceCentroidConfig):
        return module.source_centroid_config(
            temperature=config.temperature,
            use_diagonal_scale=config.use_diagonal_scale,
            shrinkage=config.shrinkage,
            epsilon=config.epsilon,
        )
    if isinstance(config, Mapping):
        return module.source_centroid_config(**dict(config))
    return _ORIGINAL_COERCE_CONFIG(config)


def _source_temperature_grid(module: Any, temperatures: Any) -> tuple[float, ...]:
    if isinstance(temperatures, str):
        raw_values = tuple(part.strip() for part in temperatures.replace(";", ",").split(",") if part.strip())
    else:
        try:
            raw_values = tuple(temperatures)
        except TypeError as exc:
            raise ValueError("temperatures must contain positive finite values.") from exc
    values = tuple(module._positive_float(value, name="temperatures") for value in raw_values)
    if not values:
        raise ValueError("temperatures must contain at least one value.")
    return values


def _numeric_scalar(value: Any, *, message: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        raise ValueError(message)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc


def _positive_float(value: Any, *, name: str) -> float:
    message = f"{name} must be positive and finite."
    parsed = _numeric_scalar(value, message=message)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(message)
    return parsed


def _unit_interval_float(value: Any, *, name: str) -> float:
    message = f"{name} must be in [0, 1]."
    parsed = _numeric_scalar(value, message=message)
    if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(message)
    return parsed


__all__ = ["install"]
