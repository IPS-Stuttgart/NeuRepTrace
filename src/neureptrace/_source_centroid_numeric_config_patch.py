"""Runtime patch for source-centroid numeric scalar configuration validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

_ORIGINAL_COERCE_CONFIG = None
_INSTALLED = False


def install() -> None:
    """Reject array-valued numeric source-centroid config controls."""
    global _INSTALLED, _ORIGINAL_COERCE_CONFIG
    if _INSTALLED:
        return

    from neureptrace.decoding import source_centroid as module

    _ORIGINAL_COERCE_CONFIG = module._coerce_config
    module.source_centroid_config = _patched_source_centroid_config
    module._coerce_config = _patched_coerce_config
    _INSTALLED = True


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
