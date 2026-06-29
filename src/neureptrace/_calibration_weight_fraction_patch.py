"""Runtime patches for calibration validation and sample-weight fractions."""

from __future__ import annotations

from functools import wraps

import numpy as np


_FRACTION_GROUP_COLUMNS = ("decoder", "emission_mode", "time")
_PATCH_ATTR = "_calibration_weight_fraction_patched"


def _weight_fraction_group_columns(frame) -> list[str]:
    return [column for column in _FRACTION_GROUP_COLUMNS if column in frame.columns]


def _per_group_weight_fractions(frame, weight_column: str, group_columns: list[str]) -> np.ndarray:
    weights = frame[weight_column].astype(float)
    if group_columns:
        totals = frame.groupby(group_columns, sort=False)[weight_column].transform("sum").astype(float)
    else:
        totals = np.full(len(frame), float(weights.sum()), dtype=float)

    weight_values = weights.to_numpy(dtype=float)
    total_values = np.asarray(totals, dtype=float)
    fractions = np.zeros(len(frame), dtype=float)
    positive = total_values > 0.0
    if np.any(positive):
        fractions[positive] = weight_values[positive] / total_values[positive]
    return fractions


def install() -> None:
    """Patch calibration validation and weight-fraction aggregation."""
    from . import _calibration_bool_numeric_patch
    import neureptrace.calibration as calibration

    _calibration_bool_numeric_patch.install()

    if getattr(calibration.aggregate_reliability_bins, _PATCH_ATTR, False):
        return

    original_aggregate_reliability_bins = calibration.aggregate_reliability_bins

    @wraps(original_aggregate_reliability_bins)
    def aggregate_reliability_bins(csv_paths):
        aggregated = original_aggregate_reliability_bins(csv_paths)
        weight_column = calibration.RELIABILITY_BIN_WEIGHT_COLUMN
        if aggregated.empty or weight_column not in aggregated.columns:
            return aggregated

        aggregated = aggregated.copy()
        group_columns = _weight_fraction_group_columns(aggregated)
        aggregated["sample_weight_fraction"] = _per_group_weight_fractions(
            aggregated,
            weight_column,
            group_columns,
        )
        return aggregated

    setattr(aggregate_reliability_bins, _PATCH_ATTR, True)
    calibration.aggregate_reliability_bins = aggregate_reliability_bins


__all__ = ["install"]
