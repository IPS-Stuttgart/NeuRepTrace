"""Validate source-alignment target-calibration seed values."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

_PATCH_MARKER = "_neureptrace_source_alignment_target_seed_patch_installed"


def install() -> None:
    """Patch source-alignment config so calibration seeds are valid RNG seeds."""

    source_alignment = importlib.import_module("neureptrace.decoding.source_alignment")
    original_config = source_alignment.source_alignment_config
    if getattr(original_config, _PATCH_MARKER, False):
        return

    @wraps(original_config)
    def source_alignment_config(
        *,
        method: str | None = None,
        anchor_mode: str | None = None,
        anchor_column: str | None = None,
        repetition_cap: int | str | None = source_alignment.DEFAULT_ALIGNMENT_REPETITION_CAP,
        components: int | float | str | None = source_alignment.DEFAULT_ALIGNMENT_COMPONENTS,
        times: Any = None,
        target_projection: str | None = "group_projection",
        target_calibration_per_anchor: int | str | None = 1,
        target_calibration_seed: int | str = 13,
        hyperalignment_iterations: int = 10,
        mcca_regularization: float = 1e-6,
        mcca_subject_pca_components: int | float | str | None = None,
    ):
        seed = source_alignment._normalize_integer(
            target_calibration_seed,
            name="alignment_target_calibration_seed",
            minimum=0,
        )
        return original_config(
            method=method,
            anchor_mode=anchor_mode,
            anchor_column=anchor_column,
            repetition_cap=repetition_cap,
            components=components,
            times=times,
            target_projection=target_projection,
            target_calibration_per_anchor=target_calibration_per_anchor,
            target_calibration_seed=seed,
            hyperalignment_iterations=hyperalignment_iterations,
            mcca_regularization=mcca_regularization,
            mcca_subject_pca_components=mcca_subject_pca_components,
        )

    setattr(source_alignment_config, _PATCH_MARKER, True)
    source_alignment.source_alignment_config = source_alignment_config


__all__ = ["install"]
