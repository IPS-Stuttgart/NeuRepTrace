"""Match OpenNeuro comparison context keys robustly after CSV round trips.

Pandas reads blank CSV fields as missing values by default.  The OpenNeuro
alignment comparison can therefore see context columns such as
``label_shuffle_seed`` as ``NaN`` after ``alignment_variant_summary.csv`` has been
saved and reloaded.  Tuple keys containing missing values do not compare equal,
so target-calibrated rows could lose their matched raw baseline.
"""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

_PATCH_MARKER = "_neureptrace_openneuro_alignment_compare_missing_context_patch_installed"


def _is_missing(value: Any) -> bool:
    """Return true for scalar pandas/numpy missing values."""

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _context_key_value(value: Any) -> Any:
    """Normalize missing group-key values without changing real values."""

    return "" if _is_missing(value) else value


def _context_key(values: Any) -> tuple[Any, ...]:
    """Return a tuple suitable for dictionary lookup across CSV round trips."""

    if not isinstance(values, tuple):
        values = (values,)
    return tuple(_context_key_value(value) for value in values)


def install() -> None:
    """Patch target-calibrated comparison raw-baseline matching."""

    module = importlib.import_module("neureptrace.openneuro_alignment_compare")
    if getattr(module, _PATCH_MARKER, False):
        return

    original = module.build_target_calibrated_comparison

    @wraps(original)
    def build_target_calibrated_comparison(variants: pd.DataFrame, *, min_delta: float = 0.0) -> pd.DataFrame:
        """Compare disjoint target-calibrated projection against strict and raw rows."""

        if variants.empty:
            return pd.DataFrame(columns=module.TARGET_CALIBRATED_COMPARISON_COLUMNS)
        rows: list[dict[str, Any]] = []
        raw_groups = {
            _context_key(group_values): group
            for group_values, group in module._valid_raw_rows(variants).groupby(module.COMPARISON_CONTEXT_COLUMNS, dropna=False)
        }
        group_columns, grouped = module._groupby_context(variants, ["alignment_method", "alignment_anchor_mode"])
        for group_values, group in grouped:
            group_map = dict(zip(group_columns, _context_key(group_values), strict=False))
            target_rows = group[group["alignment_target_projection"] == module.TARGET_CALIBRATED_TARGET_PROJECTION]
            if target_rows.empty:
                continue
            strict_rows = module._valid_strict_rows(group)
            raw_key = tuple(group_map[column] for column in module.COMPARISON_CONTEXT_COLUMNS)
            raw_rows = raw_groups.get(raw_key, pd.DataFrame())
            target_row = module._best_row(target_rows)
            strict_row = module._best_row(strict_rows) if not strict_rows.empty else None
            raw_row = module._best_row(raw_rows) if not raw_rows.empty else None
            delta_vs_strict = "" if strict_row is None else float(target_row["selection_score"]) - float(strict_row["selection_score"])
            delta_vs_raw = "" if raw_row is None else float(target_row["selection_score"]) - float(raw_row["selection_score"])
            if delta_vs_strict == "":
                decision = "target_calibrated_without_strict_pair"
                interpretation = "strict_source_only_pair_missing"
            elif float(delta_vs_strict) > min_delta:
                decision = "target_calibrated_beats_strict_source_only"
                interpretation = "small_target_calibration_can_help_this_alignment"
            elif float(delta_vs_strict) < -min_delta:
                decision = "target_calibrated_hurts_strict_source_only"
                interpretation = "target_calibration_not_sufficient_for_this_alignment"
            else:
                decision = "no_clear_target_calibration_gain_over_strict"
                interpretation = "target_calibration_inconclusive"
            rows.append(
                {
                    **group_map,
                    "strict_artifact": "" if strict_row is None else strict_row["artifact_name"],
                    "strict_value": "" if strict_row is None else strict_row["selection_value"],
                    "target_calibrated_artifact": target_row["artifact_name"],
                    "target_calibrated_value": target_row["selection_value"],
                    "raw_artifact": "" if raw_row is None else raw_row["artifact_name"],
                    "raw_value": "" if raw_row is None else raw_row["selection_value"],
                    "score_delta_target_calibrated_minus_strict": delta_vs_strict,
                    "score_delta_target_calibrated_minus_raw": delta_vs_raw,
                    "min_delta": float(min_delta),
                    "decision": decision,
                    "interpretation": interpretation,
                }
            )
        return pd.DataFrame(rows, columns=module.TARGET_CALIBRATED_COMPARISON_COLUMNS)

    module.build_target_calibrated_comparison = build_target_calibrated_comparison
    setattr(module, _PATCH_MARKER, True)


__all__ = ["install"]
