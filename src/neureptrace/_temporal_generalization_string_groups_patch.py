"""Normalize temporal-generalization summary grouping and boolean metadata."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

_PATCH_MARKER = "_neureptrace_temporal_generalization_string_groups_patch_installed"
_TRUE_BOOL_TEXT = {"1", "true", "t", "yes", "y", "on"}
_FALSE_BOOL_TEXT = {"0", "false", "f", "no", "n", "off", ""}


def _normalize_group_columns(group_columns: Sequence[str] | str | None) -> list[str]:
    if group_columns is None:
        return []
    if isinstance(group_columns, str):
        return [group_columns]
    return list(dict.fromkeys(group_columns))


def _is_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _parse_bool(value: object, *, name: str) -> bool:
    if _is_missing(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        numeric = int(value)
        if numeric in {0, 1}:
            return bool(numeric)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and numeric in {0.0, 1.0}:
            return bool(int(numeric))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_BOOL_TEXT:
            return True
        if text in _FALSE_BOOL_TEXT:
            return False
    raise ValueError(f"{name} must contain boolean values, not {value!r}.")


def _bool_series(values: Any, *, name: str) -> pd.Series:
    series = pd.Series(values, copy=False)
    parsed_values = [_parse_bool(value, name=name) for value in series.to_numpy(dtype=object)]
    return pd.Series(parsed_values, index=series.index, dtype=bool)


def _coerce_is_diagonal(frame: Any) -> Any:
    if not isinstance(frame, pd.DataFrame) or "is_diagonal" not in frame.columns:
        return frame
    coerced = frame.copy()
    coerced["is_diagonal"] = _bool_series(coerced["is_diagonal"], name="is_diagonal")
    return coerced


def install() -> None:
    """Patch summary grouping and CSV-round-tripped boolean metadata."""

    temporal_generalization = importlib.import_module("neureptrace.decoding.temporal_generalization")
    original_summarize = temporal_generalization.summarize_temporal_generalization_matrix
    if getattr(original_summarize, _PATCH_MARKER, False):
        return

    @wraps(original_summarize)
    def summarize_temporal_generalization_matrix(
        frame: Any,
        *,
        group_columns: Sequence[str] | str | None = (),
        accuracy_column: str = "accuracy",
        chance_column: str | None = "chance_accuracy",
    ):
        return original_summarize(
            _coerce_is_diagonal(frame),
            group_columns=_normalize_group_columns(group_columns),
            accuracy_column=accuracy_column,
            chance_column=chance_column,
        )

    setattr(summarize_temporal_generalization_matrix, _PATCH_MARKER, True)
    temporal_generalization.summarize_temporal_generalization_matrix = summarize_temporal_generalization_matrix


__all__ = ["install"]
