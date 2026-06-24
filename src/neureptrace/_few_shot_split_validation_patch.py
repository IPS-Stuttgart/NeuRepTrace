"""Runtime patch for stricter few-shot manual split-index validation."""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np


_INDEX_ERROR = "{name} must contain integer row indices."
_BOOLEAN_INDEX_ERROR = "{name} must contain integer row indices, not booleans or a boolean mask."


def _normalize_manual_split_indices(values: Sequence[int] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    flat = array.reshape(-1)
    if flat.dtype == np.bool_ or any(isinstance(value, (bool, np.bool_)) for value in flat.tolist()):
        raise ValueError(_BOOLEAN_INDEX_ERROR.format(name=name))
    try:
        numeric = np.asarray(flat, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(_INDEX_ERROR.format(name=name)) from exc
    if not np.all(np.isfinite(numeric)) or not np.all(numeric % 1.0 == 0.0):
        raise ValueError(_INDEX_ERROR.format(name=name))
    return numeric.astype(int, copy=False)


def install() -> None:
    """Install stricter validation for caller-provided few-shot split indices."""
    import neureptrace.decoding.few_shot as few_shot

    original_fit = few_shot.fit_few_shot_target_calibrated_decoder
    if getattr(original_fit, "_few_shot_split_validation_patched", False):
        return

    @wraps(original_fit)
    def fit_few_shot_target_calibrated_decoder(*args: Any, **kwargs: Any) -> Any:
        split = kwargs.get("split")
        if split is not None:
            kwargs = dict(kwargs)
            kwargs["split"] = few_shot.FewShotTargetCalibrationSplit(
                evaluation_indices=_normalize_manual_split_indices(split.evaluation_indices, name="evaluation_indices"),
                calibration_indices=_normalize_manual_split_indices(split.calibration_indices, name="calibration_indices"),
            )
        return original_fit(*args, **kwargs)

    fit_few_shot_target_calibrated_decoder._few_shot_split_validation_patched = True  # type: ignore[attr-defined]
    few_shot.fit_few_shot_target_calibrated_decoder = fit_few_shot_target_calibrated_decoder


__all__ = ["install"]
