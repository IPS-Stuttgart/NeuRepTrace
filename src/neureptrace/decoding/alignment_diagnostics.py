"""Diagnostics for row-aligned cross-subject alignment experiments.

These helpers are intentionally independent from the M-CCA/hyperalignment fitting
code so workflow and smoke-test code can report whether an alignment comparison
is rank-limited, calibration-free, or using a cross-window projection adapter.
That information is critical when interpreting negative alignment results: a
three-class ``class_mean`` anchor set can only support a very low-dimensional
common space and should not be compared naively with a high-dimensional
no-alignment baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import warnings

import numpy as np


LOW_RANK_CLASS_MEAN_COMPONENT_THRESHOLD = 2
TARGET_PROJECTION_GROUP_FALLBACKS = {
    "group_projection",
    "average_projection",
    "source_group_projection",
    "calibration_free_group_projection",
}


class AlignmentDiagnosticWarning(RuntimeWarning):
    """Warning emitted when an alignment configuration is likely misleading."""


@dataclass(frozen=True)
class AlignmentDiagnostics:
    """Small serializable summary for an alignment fit or smoke run."""

    method: str
    sample_mode: str
    n_alignment_rows: int
    requested_components: int
    actual_components: int | None = None
    n_classes: int | None = None
    target_projection_kind: str | None = None
    target_calibration_rows: int | None = None
    cross_window_adapter_used: bool | None = None

    @property
    def row_rank_cap(self) -> int:
        """Maximum useful centered row rank for row-aligned anchors."""

        return max(int(self.n_alignment_rows) - 1, 0)

    @property
    def effective_components(self) -> int:
        """Actual retained dimensions, or the row-rank cap if not yet fitted."""

        if self.actual_components is not None:
            return int(self.actual_components)
        return min(int(self.requested_components), self.row_rank_cap)

    @property
    def is_low_rank_class_mean(self) -> bool:
        return self.sample_mode == "class_mean" and self.row_rank_cap <= LOW_RANK_CLASS_MEAN_COMPONENT_THRESHOLD

    @property
    def uses_group_projection_fallback(self) -> bool:
        if self.target_projection_kind is None:
            return False
        normalized = str(self.target_projection_kind).strip().lower().replace("-", "_")
        return normalized in TARGET_PROJECTION_GROUP_FALLBACKS

    @property
    def has_target_calibration(self) -> bool:
        if self.target_calibration_rows is None:
            return False
        return int(self.target_calibration_rows) > 0 and not self.uses_group_projection_fallback

    def to_record(self) -> dict[str, Any]:
        """Return CSV/JSON-friendly diagnostic fields."""

        record = asdict(self)
        record["row_rank_cap"] = self.row_rank_cap
        record["effective_components"] = self.effective_components
        record["is_low_rank_class_mean"] = self.is_low_rank_class_mean
        record["uses_group_projection_fallback"] = self.uses_group_projection_fallback
        record["has_target_calibration"] = self.has_target_calibration
        return record


def requested_component_count(n_components: int | float) -> int:
    """Normalize component requests for diagnostics without importing fit internals."""

    if n_components == float("inf"):
        return int(np.iinfo(np.int32).max)

    try:
        value = float(n_components)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_components must be a positive integer component count or infinity.") from exc

    if not np.isfinite(value):
        raise ValueError("n_components must be a positive integer component count or infinity.")
    if not value.is_integer():
        raise ValueError(
            "n_components must be an integer component count or infinity; "
            "fractional variance-ratio requests are not supported for alignment components."
        )

    requested = int(value)
    if requested < 1:
        raise ValueError("n_components must be a positive integer component count or infinity.")
    return requested


def make_alignment_diagnostics(
    *,
    method: str,
    sample_mode: str,
    n_alignment_rows: int,
    n_components: int | float,
    actual_components: int | None = None,
    n_classes: int | None = None,
    target_projection_kind: str | None = None,
    target_calibration_rows: int | None = None,
    cross_window_adapter_used: bool | None = None,
) -> AlignmentDiagnostics:
    """Build a normalized diagnostic object for an alignment configuration."""

    rows = int(n_alignment_rows)
    if rows < 1:
        raise ValueError("n_alignment_rows must be positive.")
    normalized_mode = str(sample_mode).strip().lower().replace("-", "_")
    requested = requested_component_count(n_components)
    if actual_components is not None and int(actual_components) < 1:
        raise ValueError("actual_components must be positive when provided.")
    if n_classes is not None and int(n_classes) < 1:
        raise ValueError("n_classes must be positive when provided.")
    if target_calibration_rows is not None and int(target_calibration_rows) < 0:
        raise ValueError("target_calibration_rows must be non-negative when provided.")
    return AlignmentDiagnostics(
        method=str(method),
        sample_mode=normalized_mode,
        n_alignment_rows=rows,
        requested_components=requested,
        actual_components=None if actual_components is None else int(actual_components),
        n_classes=None if n_classes is None else int(n_classes),
        target_projection_kind=None if target_projection_kind is None else str(target_projection_kind),
        target_calibration_rows=None if target_calibration_rows is None else int(target_calibration_rows),
        cross_window_adapter_used=None if cross_window_adapter_used is None else bool(cross_window_adapter_used),
    )


def warn_for_alignment_diagnostics(diagnostics: AlignmentDiagnostics) -> None:
    """Emit warnings for alignment setups that can lead to misleading negatives."""

    if diagnostics.is_low_rank_class_mean:
        warnings.warn(
            (
                f"{diagnostics.method} class_mean alignment has only {diagnostics.n_alignment_rows} "
                f"anchor rows and a centered row-rank cap of {diagnostics.row_rank_cap}. "
                "For 2- or 3-class tasks this is a severe bottleneck; compare against a "
                "dimension-matched no-alignment baseline or use richer anchors such as "
                "class_repetition, pseudotrials, or stimulus-identity anchors."
            ),
            AlignmentDiagnosticWarning,
            stacklevel=2,
        )
    if diagnostics.uses_group_projection_fallback and not diagnostics.has_target_calibration:
        warnings.warn(
            (
                f"{diagnostics.method} is using target projection kind "
                f"{diagnostics.target_projection_kind!r} without target calibration rows. "
                "This is a calibration-free fallback, not the target-calibrated alignment "
                "protocol usually meant by M-CCA/hyperalignment comparisons."
            ),
            AlignmentDiagnosticWarning,
            stacklevel=2,
        )
    if diagnostics.cross_window_adapter_used:
        warnings.warn(
            (
                f"{diagnostics.method} used a cross-window projection adapter. "
                "Alignment and decoding windows should be reported separately because "
                "collapsing a projection to channel space can remove time-specific geometry."
            ),
            AlignmentDiagnosticWarning,
            stacklevel=2,
        )


__all__ = [
    "AlignmentDiagnostics",
    "AlignmentDiagnosticWarning",
    "make_alignment_diagnostics",
    "requested_component_count",
    "warn_for_alignment_diagnostics",
]
