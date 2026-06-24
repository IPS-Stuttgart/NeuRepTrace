"""Runtime guardrails for decoder classifier hyperparameters."""

from __future__ import annotations

from typing import Any

import numpy as np


def _strict_positive_float_classifier_param(
    classifier_param: Any,
    *,
    default: float,
    name: str,
) -> float:
    """Normalize positive float classifier parameters without bool coercion."""

    if classifier_param is None:
        value = float(default)
    else:
        if isinstance(classifier_param, (bool, np.bool_)):
            raise ValueError(f"{name} must be numeric, not boolean.")
        try:
            value = float(classifier_param)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive finite value.") from exc

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
    return value


def install() -> None:
    """Install strict classifier-parameter validation in ``neureptrace.decoding``."""

    from neureptrace import decoding as _decoding

    if getattr(_decoding, "_positive_float_classifier_param", None) is _strict_positive_float_classifier_param:
        return
    _decoding._positive_float_classifier_param = _strict_positive_float_classifier_param
