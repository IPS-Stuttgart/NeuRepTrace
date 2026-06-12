"""Runtime patch for composite source-alignment anchor vectors.

Some alignment anchor paths accept metadata-derived composite keys, for example
``(run, stimulus_id)`` tuples.  ``np.asarray(..., dtype=object).reshape(-1)``
still flattens rectangular tuple sequences into individual tuple elements, so a
valid one-anchor-per-trial vector can look twice as long as the feature matrix.
The core M-CCA / hyperalignment helpers already preserve such labels; this patch
brings the source-alignment guardrail and diagnostics path in line with them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_alignment_anchor_patch_installed"


def _object_vector(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Return a 1D object vector while preserving tuple-like anchors."""

    if isinstance(values, np.ndarray):
        if values.ndim == 1:
            return values.astype(object, copy=False).reshape(-1)
        rows = [tuple(np.asarray(row, dtype=object).reshape(-1).tolist()) for row in values]
    else:
        try:
            rows = list(values)
        except TypeError:
            rows = [values]

    vector = np.empty(len(rows), dtype=object)
    vector[:] = rows
    return vector


def install() -> None:
    """Install composite-anchor preservation for source-alignment helpers."""

    from neureptrace.decoding import source_alignment

    if getattr(source_alignment, _PATCH_MARKER, False):
        return

    def _anchor_vector(values: Sequence[Any] | np.ndarray | None, *, expected_length: int, name: str) -> np.ndarray:
        if values is None:
            raise ValueError(f"{name} is required for this alignment mode.")
        vector = _object_vector(values)
        if vector.shape[0] != expected_length:
            raise ValueError(f"{name} must have the same row count as the corresponding feature matrix.")
        if any(source_alignment._is_missing_anchor_value(value) for value in vector):
            raise ValueError(f"{name} contains missing values.")
        return vector

    def _update_projection_anchor_availability(
        row: dict[str, Any],
        *,
        prefix: str,
        projection_anchors: Sequence[Any] | np.ndarray | None,
        common_anchors: np.ndarray,
        failures: list[str],
        required_repetitions_per_anchor: int | None = None,
    ) -> None:
        used_key = f"{prefix}_anchor_values_used"
        rows_key = f"n_{prefix}_rows"
        values_key = f"n_{prefix}_anchor_values"
        missing_count_key = f"{prefix}_missing_common_anchor_count"
        missing_preview_key = f"{prefix}_missing_common_anchor_values_preview"
        row[used_key] = projection_anchors is not None
        if projection_anchors is None:
            failures.append(f"{prefix}_projection_missing_anchor_values")
            return

        vector = _object_vector(projection_anchors)
        row[rows_key] = int(vector.shape[0])
        missing_mask = np.asarray([source_alignment._is_missing_anchor_value(value) for value in vector], dtype=bool)
        if np.any(missing_mask):
            failures.append(f"{prefix}_projection_contains_missing_anchor_values")
        valid_vector = vector[~missing_mask]
        available = source_alignment._ordered_unique_anchor_values(valid_vector)
        row[values_key] = int(available.size)
        missing = np.asarray(
            [anchor for anchor in common_anchors if not source_alignment._contains_anchor_value(available, anchor)],
            dtype=object,
        )
        row[missing_count_key] = int(missing.size)
        row[missing_preview_key] = source_alignment._preview_values(missing)
        if missing.size:
            failures.append(f"{prefix}_subject_missing_alignment_anchors")
        if required_repetitions_per_anchor is not None and required_repetitions_per_anchor > 1:
            insufficient = np.asarray(
                [
                    anchor
                    for anchor in common_anchors
                    if source_alignment._count_anchor_value(valid_vector, anchor) < int(required_repetitions_per_anchor)
                ],
                dtype=object,
            )
            if insufficient.size:
                failures.append(f"{prefix}_subject_insufficient_alignment_anchor_repetitions")
                row["prefit_failure_detail"] = (
                    f"{prefix} anchors require at least {int(required_repetitions_per_anchor)} repetition(s) "
                    f"per common anchor; insufficient anchors: {source_alignment._preview_values(insufficient)}"
                )

    source_alignment._anchor_vector = _anchor_vector
    source_alignment._update_projection_anchor_availability = _update_projection_anchor_availability
    setattr(source_alignment, _PATCH_MARKER, True)
