from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

Window = tuple[float, float]


def summarize_window_metric(
    frame: pd.DataFrame,
    metric_column: str,
    window: Window,
    time_column: str = "time",
    group_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Summarize one metric inside an inclusive time window."""
    group_columns = _normalize_columns(group_columns)
    _validate_group_output_columns(
        group_columns,
        _summary_output_columns(metric_column),
        operation="summary",
    )
    _require_columns(frame, [time_column, metric_column, *group_columns])
    window_start, window_stop = _validate_window(window)
    time_values = _finite_numeric_series(frame[time_column], name=time_column)

    window_frame = frame.loc[(time_values >= window_start) & (time_values <= window_stop)]
    if window_frame.empty:
        raise ValueError(f"No rows fall inside window [{window_start}, {window_stop}].")

    rows: list[dict[str, object]] = []
    for group_key, group in _iter_groups(window_frame, group_columns):
        row = _group_row(group_columns, group_key)
        values = _finite_numeric_or_missing_series(group[metric_column], name=metric_column)
        row.update(
            {
                "window_start": window_start,
                "window_stop": window_stop,
                "n_rows": int(values.notna().sum()),
                f"{metric_column}_mean": _float_or_nan(values.mean()),
                f"{metric_column}_std": _float_or_nan(values.std()),
                f"{metric_column}_sem": _float_or_nan(values.sem()),
            }
        )
        rows.append(row)

    return _sorted_frame(rows, group_columns)


def compare_prepost_windows(
    frame: pd.DataFrame,
    metric_column: str,
    pre_window: Window,
    post_window: Window,
    time_column: str = "time",
    group_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Compare a metric between inclusive pre and post time windows."""
    group_columns = _normalize_columns(group_columns)
    _validate_group_output_columns(
        group_columns,
        _comparison_output_columns(metric_column),
        operation="comparison",
    )
    pre_summary = summarize_window_metric(frame, metric_column, pre_window, time_column=time_column, group_columns=group_columns)
    post_summary = summarize_window_metric(frame, metric_column, post_window, time_column=time_column, group_columns=group_columns)

    pre_summary = pre_summary.rename(
        columns={
            "window_start": "pre_window_start",
            "window_stop": "pre_window_stop",
            "n_rows": "n_pre_rows",
            f"{metric_column}_mean": f"{metric_column}_pre_mean",
            f"{metric_column}_std": f"{metric_column}_pre_std",
            f"{metric_column}_sem": f"{metric_column}_pre_sem",
        }
    )
    post_summary = post_summary.rename(
        columns={
            "window_start": "post_window_start",
            "window_stop": "post_window_stop",
            "n_rows": "n_post_rows",
            f"{metric_column}_mean": f"{metric_column}_post_mean",
            f"{metric_column}_std": f"{metric_column}_post_std",
            f"{metric_column}_sem": f"{metric_column}_post_sem",
        }
    )

    if group_columns:
        merged = pre_summary.merge(post_summary, on=group_columns, how="outer")
    else:
        merged = pd.concat([pre_summary.reset_index(drop=True), post_summary.reset_index(drop=True)], axis=1)
    merged[f"{metric_column}_post_minus_pre"] = merged[f"{metric_column}_post_mean"] - merged[f"{metric_column}_pre_mean"]
    return _sorted_frame(merged.to_dict("records"), group_columns)


def _normalize_columns(columns: Sequence[str] | str | None) -> list[str]:
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    return list(dict.fromkeys(columns))


def _summary_output_columns(metric_column: str) -> set[str]:
    return {
        "window_start",
        "window_stop",
        "n_rows",
        f"{metric_column}_mean",
        f"{metric_column}_std",
        f"{metric_column}_sem",
    }


def _comparison_output_columns(metric_column: str) -> set[str]:
    return {
        "pre_window_start",
        "pre_window_stop",
        "n_pre_rows",
        f"{metric_column}_pre_mean",
        f"{metric_column}_pre_std",
        f"{metric_column}_pre_sem",
        "post_window_start",
        "post_window_stop",
        "n_post_rows",
        f"{metric_column}_post_mean",
        f"{metric_column}_post_std",
        f"{metric_column}_post_sem",
        f"{metric_column}_post_minus_pre",
    }


def _validate_group_output_columns(
    group_columns: Sequence[str],
    output_columns: set[str],
    *,
    operation: str,
) -> None:
    collisions = sorted(set(group_columns).intersection(output_columns))
    if collisions:
        raise ValueError(
            f"group_columns overlap generated {operation} columns: {collisions}"
        )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    required = list(dict.fromkeys(columns))
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Data frame is missing required columns: {missing}")
    frame_columns = frame.columns.tolist()
    ambiguous = [column for column in required if frame_columns.count(column) > 1]
    if ambiguous:
        raise ValueError(f"Data frame has ambiguous duplicate required columns: {ambiguous}")


def _validate_window(window: Window) -> Window:
    if isinstance(window, (str, bytes, bytearray)):
        raise ValueError("window must contain exactly two values")
    try:
        window_length = len(window)
    except TypeError as exc:
        raise ValueError("window must contain exactly two values") from exc
    if window_length != 2:
        raise ValueError("window must contain exactly two values")
    window_start = _validate_window_endpoint(window[0], name="start")
    window_stop = _validate_window_endpoint(window[1], name="stop")
    if window_start > window_stop:
        raise ValueError("window start must be less than or equal to window stop")
    return window_start, window_stop


def _validate_window_endpoint(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"window {name} must be a finite numeric value")
    if isinstance(value, np.ndarray):
        raise ValueError(f"window {name} must be a finite numeric scalar")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"window {name} must be a finite numeric value") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"window {name} must be a finite numeric value")
    return numeric


def _finite_numeric_series(values: object, *, name: str) -> pd.Series:
    series = pd.Series(values)
    if series.map(_is_boolean_scalar).any():
        raise ValueError(f"{name} must contain only finite numeric values")
    parsed = pd.to_numeric(series, errors="coerce")
    numeric = parsed.to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return parsed


def _finite_numeric_or_missing_series(values: object, *, name: str) -> pd.Series:
    series = pd.Series(values)
    message = f"{name} must contain only finite numeric values or missing values"
    if series.map(_is_boolean_scalar).any():
        raise ValueError(message)
    parsed = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & parsed.isna()
    numeric = parsed.dropna().to_numpy(dtype=float)
    if bool(invalid.any()) or not np.all(np.isfinite(numeric)):
        raise ValueError(message)
    return parsed


def _is_boolean_scalar(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _iter_groups(frame: pd.DataFrame, group_columns: Sequence[str]):
    if not group_columns:
        yield (), frame
        return
    yield from frame.groupby(list(group_columns), dropna=False, sort=True)


def _group_row(group_columns: Sequence[str], group_key: object) -> dict[str, object]:
    if not group_columns:
        return {}
    if len(group_columns) == 1 and not isinstance(group_key, tuple):
        group_key = (group_key,)
    return dict(zip(group_columns, group_key))


def _sorted_frame(rows: list[dict[str, object]], group_columns: Sequence[str]) -> pd.DataFrame:
    result = pd.DataFrame(rows)
    if group_columns and not result.empty:
        try:
            result = result.sort_values(list(group_columns), kind="mergesort")
        except TypeError:
            # Group identifiers are labels and need not define a shared ordering
            # across Python types. Grouping and merging already provide a stable
            # row order, so retain it when the cosmetic sort is undefined.
            pass
    return result.reset_index(drop=True)


def _float_or_nan(value: object) -> float:
    return float(value) if pd.notna(value) else float("nan")
