"""Runtime patch for confidence-selection config normalization."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_ATTR = "_confidence_selection_direct_config_normalization_patch"
_SCALAR_PATCH_ATTR = "_confidence_selection_scalar_validation_patch"


def _normalize_config(config: Any) -> Any:
    import neureptrace.decoding.confidence_selection as confidence_selection

    if isinstance(config, confidence_selection.ConfidenceSelectionConfig):
        return confidence_selection.confidence_selection_config(
            mode=config.mode,
            threshold=config.threshold,
            top_k=config.top_k,
            per_class_top_k=config.per_class_top_k,
            min_margin=config.min_margin,
            epsilon=config.epsilon,
        )
    return confidence_selection.confidence_selection_config(**dict(config))


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be an integer.")
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    return int(numeric)


def _float_value(value: Any, *, name: str) -> float:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be finite.")
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    return numeric


def install() -> None:
    """Install config normalization and scalar guards."""
    import neureptrace.decoding.confidence_selection as confidence_selection

    if not getattr(confidence_selection, _SCALAR_PATCH_ATTR, False):
        confidence_selection._integer = _integer
        confidence_selection._float_value = _float_value
        setattr(confidence_selection, _SCALAR_PATCH_ATTR, True)

    original_select_confident_probability_rows = confidence_selection.select_confident_probability_rows
    if not getattr(original_select_confident_probability_rows, _PATCH_ATTR, False):

        @wraps(original_select_confident_probability_rows)
        def select_confident_probability_rows(probabilities: Any, *, config: Any = None) -> Any:
            normalized_config = None if config is None else _normalize_config(config)
            return original_select_confident_probability_rows(probabilities, config=normalized_config)

        setattr(select_confident_probability_rows, _PATCH_ATTR, True)
        confidence_selection.select_confident_probability_rows = select_confident_probability_rows


__all__ = ["install"]
