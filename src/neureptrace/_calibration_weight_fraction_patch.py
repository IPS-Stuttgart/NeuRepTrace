"""Runtime patches for calibration validation and sample-weight fractions."""

from __future__ import annotations

from functools import wraps

import numpy as np


_FRACTION_GROUP_COLUMNS = ("decoder", "emission_mode", "time")
_GROUP_COLUMN_DEFAULTS = {"decoder": "overall", "emission_mode": "calibrated"}
_WEIGHT_FRACTION_PATCH_ATTR = "_calibration_weight_fraction_patched"
_EMPTY_BIN_WEIGHT_PATCH_ATTR = "_calibration_empty_bin_weight_patched"
_GROUP_METADATA_PATCH_ATTR = "_calibration_group_metadata_patched"
_SUMMARY_GROUP_METADATA_PATCH_ATTR = "_calibration_summary_group_metadata_patched"


def _weight_fraction_group_columns(frame) -> list[str]:
    return [column for column in _FRACTION_GROUP_COLUMNS if column in frame.columns]


def _per_group_weight_fractions(frame, weight_column: str, group_columns: list[str]) -> np.ndarray:
    weight_values = frame[weight_column].to_numpy(dtype=float)
    fractions = np.zeros(len(frame), dtype=float)

    if group_columns:
        grouped_positions = frame.groupby(group_columns, sort=False, dropna=False).indices.values()
    else:
        grouped_positions = (np.arange(len(frame), dtype=int),)

    for positions in grouped_positions:
        positions = np.asarray(positions, dtype=int)
        group_weights = weight_values[positions]
        max_weight = float(group_weights.max())
        if max_weight <= 0.0:
            continue

        scaled_weights = group_weights / max_weight
        scaled_total = float(scaled_weights.sum())
        fractions[positions] = scaled_weights / scaled_total

    return fractions


def _reject_positive_empty_bin_weights(frame, csv_path, weight_column: str) -> None:
    if weight_column not in frame.columns:
        return
    empty_with_weight = frame["n_samples"].eq(0) & frame[weight_column].astype(float).gt(0.0)
    if empty_with_weight.any():
        bad_rows = empty_with_weight[empty_with_weight].index.tolist()[:5]
        raise ValueError(
            f"{csv_path} contains positive {weight_column} for empty reliability bin(s) at row(s) {bad_rows}. "
            "Rows with n_samples == 0 must have zero aggregation weight."
        )


def _normalise_present_group_columns(frame):
    normalised = frame.copy()
    for column, default in _GROUP_COLUMN_DEFAULTS.items():
        if column in normalised.columns:
            normalised[column] = normalised[column].where(normalised[column].notna(), default)
    return normalised


def install() -> None:
    """Patch calibration validation and weight-fraction aggregation."""
    from . import _calibration_bool_numeric_patch
    import neureptrace.calibration as calibration

    _calibration_bool_numeric_patch.install()

    if not getattr(calibration._validate_calibration_summary, _SUMMARY_GROUP_METADATA_PATCH_ATTR, False):
        original_summary_validator = calibration._validate_calibration_summary

        @wraps(original_summary_validator)
        def _validate_calibration_summary(frame):
            validated = original_summary_validator(frame)
            return _normalise_present_group_columns(validated)

        setattr(_validate_calibration_summary, _SUMMARY_GROUP_METADATA_PATCH_ATTR, True)
        calibration._validate_calibration_summary = _validate_calibration_summary

    if not getattr(calibration._validate_reliability_bins, _EMPTY_BIN_WEIGHT_PATCH_ATTR, False):
        original_empty_bin_validator = calibration._validate_reliability_bins

        @wraps(original_empty_bin_validator)
        def _validate_reliability_bins(frame, csv_path):
            validated = original_empty_bin_validator(frame, csv_path)
            _reject_positive_empty_bin_weights(
                validated,
                csv_path,
                calibration.RELIABILITY_BIN_WEIGHT_COLUMN,
            )
            return validated

        setattr(_validate_reliability_bins, _EMPTY_BIN_WEIGHT_PATCH_ATTR, True)
        calibration._validate_reliability_bins = _validate_reliability_bins

    if not getattr(calibration._validate_reliability_bins, _GROUP_METADATA_PATCH_ATTR, False):
        original_group_metadata_validator = calibration._validate_reliability_bins

        @wraps(original_group_metadata_validator)
        def _validate_reliability_bin_groups(frame, csv_path):
            validated = original_group_metadata_validator(frame, csv_path)
            return _normalise_present_group_columns(validated)

        setattr(_validate_reliability_bin_groups, _GROUP_METADATA_PATCH_ATTR, True)
        calibration._validate_reliability_bins = _validate_reliability_bin_groups

    if not getattr(calibration.aggregate_reliability_bins, _WEIGHT_FRACTION_PATCH_ATTR, False):
        original_aggregate_reliability_bins = calibration.aggregate_reliability_bins

        @wraps(original_aggregate_reliability_bins)
        def aggregate_reliability_bins(csv_paths):
            # The original implementation computes a global weight fraction that
            # this patch replaces below. Suppress overflow from that discarded
            # intermediate when individually finite per-bin weights have a total
            # above the float64 range.
            with np.errstate(over="ignore", invalid="ignore"):
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

        setattr(aggregate_reliability_bins, _WEIGHT_FRACTION_PATCH_ATTR, True)
        calibration.aggregate_reliability_bins = aggregate_reliability_bins


__all__ = ["install"]
