"""Validate target-calibration repetition offsets before indexing rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any

import numpy as np

_INSTALLED = False


def install() -> None:
    """Install stricter validation for target repetition-offset mappings."""

    global _INSTALLED
    if _INSTALLED:
        return

    import neureptrace.decoding.source_alignment as source_alignment

    _patch_module(source_alignment)
    _INSTALLED = True


def _patch_module(module: ModuleType) -> None:
    original_normalize = module._normalize_target_repetition_offsets
    if getattr(original_normalize, "__neureptrace_target_offsets_patch__", False):
        return

    def _normalize_target_repetition_offsets(
        selected_offsets_by_class: Mapping[int, Sequence[int] | np.ndarray] | None,
        *,
        labels: np.ndarray,
        classes: np.ndarray,
        n_repetitions_per_class: int,
    ) -> dict[int, np.ndarray] | None:
        if selected_offsets_by_class is None:
            return original_normalize(
                selected_offsets_by_class,
                labels=labels,
                classes=classes,
                n_repetitions_per_class=n_repetitions_per_class,
            )

        validated_offsets_by_class: dict[int, np.ndarray] = {}
        for class_position, _class_label in enumerate(classes):
            try:
                raw_offsets = selected_offsets_by_class[class_position]
            except KeyError as exc:
                raise ValueError(f"selected_offsets_by_class is missing class position {class_position}.") from exc

            offsets = np.asarray(raw_offsets)
            if offsets.ndim != 1:
                raise ValueError("selected_offsets_by_class entries must be one-dimensional.")
            if offsets.size != n_repetitions_per_class:
                raise ValueError(
                    "selected_offsets_by_class entries must match n_repetitions_per_class: "
                    f"{offsets.size} != {n_repetitions_per_class}."
                )
            if _contains_boolean(offsets):
                raise ValueError("selected_offsets_by_class entries must contain integer offsets, not booleans.")
            try:
                numeric_offsets = offsets.astype(float)
            except (TypeError, ValueError) as exc:
                raise ValueError("selected_offsets_by_class entries must contain integer offsets.") from exc
            if not np.isfinite(numeric_offsets).all() or not np.equal(numeric_offsets, np.rint(numeric_offsets)).all():
                raise ValueError("selected_offsets_by_class entries must contain finite integer offsets.")

            validated_offsets_by_class[class_position] = numeric_offsets.astype(int)

        return original_normalize(
            validated_offsets_by_class,
            labels=labels,
            classes=classes,
            n_repetitions_per_class=n_repetitions_per_class,
        )

    _normalize_target_repetition_offsets.__neureptrace_target_offsets_patch__ = True  # type: ignore[attr-defined]
    module._normalize_target_repetition_offsets = _normalize_target_repetition_offsets


def _contains_boolean(values: np.ndarray) -> bool:
    if np.issubdtype(values.dtype, np.bool_):
        return True
    return any(isinstance(value, (bool, np.bool_)) for value in values.reshape(-1))
