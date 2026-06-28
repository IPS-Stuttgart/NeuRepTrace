"""Reject boolean time-decode metric values before numeric coercion."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

_PATCH_ATTR = "_neureptrace_rejects_boolean_time_decode_metrics"


def _boolean_mask(values: pd.Series) -> pd.Series:
    return values.map(lambda value: isinstance(value, (bool, np.bool_))).fillna(False).astype(bool)


def install() -> None:
    """Install a guard for aggregate result metric columns."""

    from neureptrace import results

    original = results._coerce_finite_metric_columns
    if getattr(original, _PATCH_ATTR, False):
        return

    def _coerce_finite_metric_columns_checked(
        frame: pd.DataFrame,
        metric_columns: Sequence[str],
    ) -> pd.DataFrame:
        for metric in metric_columns:
            if metric not in frame.columns:
                continue
            boolean_values = _boolean_mask(frame[metric])
            if boolean_values.any():
                bad_rows = boolean_values[boolean_values].index.tolist()[:5]
                raise ValueError(
                    f"Metric column '{metric}' must contain finite numeric values, not booleans; "
                    f"boolean row(s): {bad_rows}."
                )
        return original(frame, metric_columns)

    setattr(_coerce_finite_metric_columns_checked, _PATCH_ATTR, True)
    _coerce_finite_metric_columns_checked.__wrapped__ = original
    results._coerce_finite_metric_columns = _coerce_finite_metric_columns_checked


__all__ = ["install"]
