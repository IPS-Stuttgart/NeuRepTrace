"""Runtime safeguards for BUSH-MEG all-protocol report helpers."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_BOOL_PATCH_ATTR = "_neureptrace_bushmeg_report_bool_like_lists"
_PATCH_ATTR = _BOOL_PATCH_ATTR
_CSV_PATCH_ATTR = "_neureptrace_bushmeg_report_empty_csv_guard"
_TRUE_STRINGS = {"1", "true", "yes", "y"}
_EMPTY_STRINGS = {"", "nan", "none", "null", "<na>"}


def _is_missing_scalar(value: Any) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _items(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, np.ndarray):
        return [value.item()] if value.ndim == 0 else list(value.ravel())
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _scalar_bool_like(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        return False if text in _EMPTY_STRINGS else text in _TRUE_STRINGS
    if value is None or _is_missing_scalar(value):
        return False
    return bool(value)


def _bool_like(value: Any) -> bool:
    values = []
    for item in _items(value):
        if isinstance(item, str):
            if item.strip().lower() in _EMPTY_STRINGS:
                continue
        elif item is None or _is_missing_scalar(item):
            continue
        values.append(item)
    return any(_scalar_bool_like(item) for item in values)


def _read_csv_or_empty(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _install_bool_patch(report: Any) -> None:
    original = getattr(report, "_bool_like", None)
    if getattr(original, _BOOL_PATCH_ATTR, False):
        return
    setattr(_bool_like, _BOOL_PATCH_ATTR, True)
    if original is not None:
        _bool_like.__wrapped__ = original
    report._bool_like = _bool_like


def _install_empty_csv_patch(report: Any) -> None:
    original = getattr(report, "_read_csv_or_empty", None)
    if getattr(original, _CSV_PATCH_ATTR, False):
        return
    setattr(_read_csv_or_empty, _CSV_PATCH_ATTR, True)
    if original is not None:
        _read_csv_or_empty.__wrapped__ = original
    report._read_csv_or_empty = _read_csv_or_empty


def install() -> None:
    report = importlib.import_module("neureptrace.bushmeg_all_protocols_report")
    _install_bool_patch(report)
    _install_empty_csv_patch(report)


__all__ = ["install"]
