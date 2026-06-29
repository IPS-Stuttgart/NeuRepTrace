"""Make BUSH-MEG all-protocol audit helpers robust to list-valued metadata."""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

_PATCH_MARKER = "_neureptrace_bushmeg_all_protocols_audit_list_values_patch_installed"
_TRUE_TOKENS = {"1", "true", "yes", "y", "on"}


def _is_nonstring_iterable(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray))


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, Mapping) or _is_nonstring_iterable(value):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _coerce_repeated_scalar(value: Any) -> Any:
    if isinstance(value, Mapping) or not _is_nonstring_iterable(value):
        return value
    items = [_coerce_repeated_scalar(item) for item in value if not _is_missing_scalar(item)]
    if not items:
        return pd.NA
    tokens = {str(item) for item in items}
    if len(tokens) == 1:
        return items[0]
    return value


def _bool_like(value: Any) -> bool:
    value = _coerce_repeated_scalar(value)
    if _is_missing_scalar(value):
        return False
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_TOKENS
    if isinstance(value, Mapping):
        return bool(value)
    if _is_nonstring_iterable(value):
        return any(_bool_like(item) for item in value)
    return bool(value)


def _first_present(row: pd.Series, names: tuple[str, ...]) -> Any:
    for name in names:
        if name not in row.index:
            continue
        value = _coerce_repeated_scalar(row[name])
        if not _is_missing_scalar(value):
            return value
    return pd.NA


def _numeric_or_na(value: Any) -> float:
    value = _coerce_repeated_scalar(value)
    if _is_missing_scalar(value):
        return float("nan")
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if not pd.isna(parsed) else float("nan")


def _format_label_value(value: Any) -> str:
    value = _coerce_repeated_scalar(value)
    if isinstance(value, Mapping):
        return str(value)
    if _is_nonstring_iterable(value):
        return "|".join(str(item) for item in value)
    return str(value)


def _row_label(row: pd.Series) -> str:
    parts = []
    for column in ("method", "outer_test_subject", "fold_index", "k_per_class", "target_calibration_per_class"):
        if column not in row.index:
            continue
        value = _coerce_repeated_scalar(row[column])
        if not _is_missing_scalar(value):
            parts.append(f"{column}={_format_label_value(value)}")
    return ", ".join(parts) if parts else f"row_index={row.name}"


def _has_text(value: Any) -> bool:
    value = _coerce_repeated_scalar(value)
    if _is_missing_scalar(value):
        return False
    if isinstance(value, Mapping):
        return bool(value)
    if _is_nonstring_iterable(value):
        return any(_has_text(item) for item in value)
    return bool(str(value).strip())


def _row_is_explicitly_skipped(row: pd.Series) -> bool:
    for column in ("target_calibration_skipped", "fold_skipped", "skipped", "is_skipped"):
        if column in row.index and _bool_like(row[column]):
            return True
    for column in ("skip_reason", "target_calibration_skip_reason", "fold_skip_reason"):
        if column in row.index and _has_text(row[column]):
            return True
    return False


def _label_tokens(value: Any) -> list[str]:
    value = _coerce_repeated_scalar(value)
    if _is_missing_scalar(value):
        return []
    if isinstance(value, Mapping):
        return []
    if _is_nonstring_iterable(value):
        tokens: list[str] = []
        for item in value:
            tokens.extend(_label_tokens(item))
        return tokens
    return [token for token in str(value).replace(",", "|").split("|") if token != ""]


def _class_count_from_row(row: pd.Series) -> float:
    if "n_classes" in row.index:
        value = _numeric_or_na(row["n_classes"])
        if not pd.isna(value):
            return value
    if "class_names" in row.index:
        labels = _label_tokens(row["class_names"])
        if labels:
            return float(len(labels))
    return float("nan")


def _parse_index_set(value: Any) -> set[int]:
    value = _coerce_repeated_scalar(value)
    if _is_missing_scalar(value):
        return set()
    if isinstance(value, Mapping):
        return set()
    if _is_nonstring_iterable(value):
        indices: set[int] = set()
        for item in value:
            indices.update(_parse_index_set(item))
        return indices
    text = str(value).strip()
    if not text:
        return set()
    for character in "[](){}":
        text = text.replace(character, " ")
    for separator in ("|", ";", ","):
        text = text.replace(separator, " ")
    indices: set[int] = set()
    for token in text.split():
        try:
            indices.add(int(float(str(token))))
        except ValueError:
            continue
    return indices


def _integer_index_or_none(value: Any) -> int | None:
    value = _coerce_repeated_scalar(value)
    if _is_missing_scalar(value) or isinstance(value, Mapping) or _is_nonstring_iterable(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric) or numeric % 1.0 != 0.0:
        return None
    return int(numeric)


def _protocol3_prediction_overlap_failures(summary: pd.DataFrame, predictions: pd.DataFrame) -> list[str]:
    audit = importlib.import_module("neureptrace.bushmeg_all_protocols_audit")
    p3_summary = audit._protocol_rows(summary, 3)
    if p3_summary.empty or predictions.empty or "protocol_category" not in predictions.columns:
        return []
    p3_predictions = audit._protocol_rows(predictions, 3)
    if p3_predictions.empty:
        return []
    index_column = next((column for column in audit.PROTOCOL3_CALIBRATION_INDEX_COLUMNS if column in p3_summary.columns), "")
    if not index_column:
        return []
    prediction_index_column = "target_row_index" if "target_row_index" in p3_predictions.columns else "trial_index" if "trial_index" in p3_predictions.columns else ""
    if not prediction_index_column:
        return ["Protocol 3 summary exposes calibration row indices, but predictions lack `target_row_index`/`trial_index` for overlap auditing."]

    calibration_by_key: dict[tuple[str, str, str, str], set[int]] = {}
    for _, row in p3_summary.iterrows():
        indices = _parse_index_set(row[index_column])
        if indices:
            calibration_by_key.setdefault(audit._protocol3_group_key(row), set()).update(indices)
    if not calibration_by_key:
        return []

    failures: list[str] = []
    for _, row in p3_predictions.iterrows():
        key = audit._protocol3_group_key(row)
        calibration_indices = calibration_by_key.get(key, set())
        prediction_value = _coerce_repeated_scalar(row.get(prediction_index_column))
        if not calibration_indices or _is_missing_scalar(prediction_value):
            continue
        row_index = _integer_index_or_none(prediction_value)
        if row_index is None:
            failures.append(
                f"Protocol 3 prediction has non-integer `{prediction_index_column}`={_format_label_value(prediction_value)!r} "
                f"for {audit._row_label(row)}."
            )
            continue
        if row_index in calibration_indices:
            failures.append(f"Protocol 3 prediction uses calibration row {row_index} for {audit._row_label(row)}.")
    return failures[:10]


def install() -> None:
    audit = importlib.import_module("neureptrace.bushmeg_all_protocols_audit")
    if getattr(audit, _PATCH_MARKER, False):
        return
    audit._bool_like = _bool_like
    audit._first_present = _first_present
    audit._numeric_or_na = _numeric_or_na
    audit._row_label = _row_label
    audit._row_is_explicitly_skipped = _row_is_explicitly_skipped
    audit._class_count_from_row = _class_count_from_row
    audit._parse_index_set = _parse_index_set
    audit._protocol3_prediction_overlap_failures = _protocol3_prediction_overlap_failures
    setattr(audit, _PATCH_MARKER, True)


__all__ = ["install"]
