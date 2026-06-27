from __future__ import annotations

from typing import Any

import numpy as np

from neureptrace.decoding import adaptive_normalization as _adaptive_normalization


def _bool_config(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
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


def install() -> None:
    _adaptive_normalization.adaptive_normalization_config = _patched_adaptive_normalization_config
