"""Preserve few-shot class order when labels cannot be sorted safely.

The few-shot target-calibrated decoder can be used without an explicit
``classes`` argument.  The base implementation then derives a class order with
``np.unique`` over the source labels plus target calibration labels.  That is
unsafe for mixed Python object labels and some composite labels because NumPy
sorts object arrays and may raise when values are not mutually orderable.

This patch injects the same observed class set in first-seen order before the
base implementation reaches that fallback.  Only source labels and target
calibration labels are considered, so evaluation labels remain score-only.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

from neureptrace._tuple_label_calibration_split_patch import (
    _atomic_label_vector,
    _normalize_manual_split_indices,
    _ordered_unique_values,
)

_PATCH_MARKER = "_neureptrace_few_shot_class_order_patch_installed"


def _observed_few_shot_class_order(
    source_labels: Sequence[Any] | np.ndarray,
    target_labels: Sequence[Any] | np.ndarray,
    split: Any,
) -> np.ndarray:
    """Return source-plus-calibration class labels without sorting objects."""

    source_vector = _atomic_label_vector(source_labels, name="source_labels")
    target_vector = _atomic_label_vector(target_labels, name="target_labels")
    calibration_indices = _normalize_manual_split_indices(split.calibration_indices, name="calibration_indices")
    if np.any(calibration_indices < 0) or np.any(calibration_indices >= target_vector.shape[0]):
        raise ValueError("calibration_indices contains an out-of-range target row index.")
    calibration_labels = target_vector[calibration_indices]
    return _ordered_unique_values(list(source_vector) + list(calibration_labels))


def install() -> None:
    """Install robust implicit class ordering for few-shot target calibration."""

    few_shot = importlib.import_module("neureptrace.decoding.few_shot")
    original_fit = few_shot.fit_few_shot_target_calibrated_decoder
    if getattr(original_fit, _PATCH_MARKER, False):
        return

    @wraps(original_fit)
    def fit_few_shot_target_calibrated_decoder(*args: Any, **kwargs: Any):
        if args or kwargs.get("classes") is not None:
            return original_fit(*args, **kwargs)
        if "source_labels" not in kwargs or "target_labels" not in kwargs:
            return original_fit(*args, **kwargs)

        call_kwargs = dict(kwargs)
        split = call_kwargs.get("split")
        if split is None:
            split = few_shot.select_few_shot_target_calibration_split(
                _atomic_label_vector(call_kwargs["target_labels"], name="target_labels"),
                per_class=call_kwargs.get("per_class", 1),
                seed=call_kwargs.get("seed", 13),
                context=call_kwargs.get("context", ()),
                min_evaluation_per_class=call_kwargs.get("min_evaluation_per_class", 1),
            )
            call_kwargs["split"] = split

        call_kwargs["classes"] = _observed_few_shot_class_order(
            call_kwargs["source_labels"],
            call_kwargs["target_labels"],
            split,
        )
        return original_fit(**call_kwargs)

    setattr(fit_few_shot_target_calibrated_decoder, _PATCH_MARKER, True)
    few_shot.fit_few_shot_target_calibrated_decoder = fit_few_shot_target_calibrated_decoder


__all__ = ["install"]
