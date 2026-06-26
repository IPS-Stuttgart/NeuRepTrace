"""Runtime patch for strict few-shot target index validation."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_few_shot_target_index_patch_installed"
_INDEX_ERROR_SUFFIX = "must contain integer row indices."
_BOOLEAN_INDEX_ERROR_SUFFIX = "must contain integer row indices, not boolean values."


def _normalize_index_vector(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} {_INDEX_ERROR_SUFFIX}")

    normalized: list[int] = []
    for value in array.tolist():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} {_BOOLEAN_INDEX_ERROR_SUFFIX}")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} {_INDEX_ERROR_SUFFIX}") from exc
        if not np.isfinite(numeric) or numeric % 1.0 != 0.0:
            raise ValueError(f"{name} {_INDEX_ERROR_SUFFIX}")
        normalized.append(int(numeric))
    return np.asarray(normalized, dtype=int)


def install() -> None:
    """Install strict validation for few-shot target index vectors."""

    from neureptrace.decoding import few_shot

    if getattr(few_shot, _PATCH_MARKER, False):
        return

    original_select = few_shot.select_few_shot_target_calibration_split
    original_fit = few_shot.fit_few_shot_target_calibrated_decoder

    @wraps(original_select)
    def select_few_shot_target_calibration_split(
        labels,
        target_indices=None,
        *,
        per_class=1,
        seed=13,
        context=(),
        min_evaluation_per_class=1,
    ):
        if target_indices is not None:
            target_indices = _normalize_index_vector(target_indices, name="target_indices")
        return original_select(
            labels,
            target_indices,
            per_class=per_class,
            seed=seed,
            context=context,
            min_evaluation_per_class=min_evaluation_per_class,
        )

    @wraps(original_fit)
    def fit_few_shot_target_calibrated_decoder(*args, **kwargs):
        if args:
            return original_fit(*args, **kwargs)
        split = kwargs.get("split")
        if split is not None:
            kwargs = dict(kwargs)
            kwargs["split"] = few_shot.FewShotTargetCalibrationSplit(
                evaluation_indices=_normalize_index_vector(
                    split.evaluation_indices,
                    name="evaluation_indices",
                ),
                calibration_indices=_normalize_index_vector(
                    split.calibration_indices,
                    name="calibration_indices",
                ),
            )
        return original_fit(**kwargs)

    few_shot.select_few_shot_target_calibration_split = select_few_shot_target_calibration_split
    few_shot.fit_few_shot_target_calibrated_decoder = fit_few_shot_target_calibrated_decoder
    setattr(few_shot, _PATCH_MARKER, True)


__all__ = ["install"]
