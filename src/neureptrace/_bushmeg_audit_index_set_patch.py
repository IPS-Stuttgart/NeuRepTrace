"""Patch calibration-index parsing and key normalization in the BUSH-MEG audit."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import pandas as pd


def _is_missing_scalar(value: Any) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, Iterable) and not isinstance(missing, (str, bytes)):
        return False
    return bool(missing)


def _parse_index_set(value: Any) -> set[int]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return set()
        for character in "[](){}":
            text = text.replace(character, " ")
        for separator in ("|", ";", ","):
            text = text.replace(separator, " ")
        tokens = text.split()
    elif _is_missing_scalar(value):
        return set()
    elif isinstance(value, Iterable):
        tokens = value
    else:
        tokens = (value,)

    indices: set[int] = set()
    for token in tokens:
        if _is_missing_scalar(token):
            continue
        try:
            indices.add(int(float(str(token))))
        except (TypeError, ValueError):
            continue
    return indices


def _first_present(row: pd.Series, names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row.index and not _is_missing_scalar(row[name]):
            return row[name]
    return pd.NA


def _canonical_text_key(value: Any) -> str:
    """Return a stable text key while preserving non-numeric string IDs."""

    if _is_missing_scalar(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not pd.isna(parsed):
        number = float(parsed)
        if math.isfinite(number):
            return str(int(number)) if number.is_integer() else f"{number:g}"
    return str(value).strip()


def _canonical_numeric_key(value: Any) -> str:
    """Canonicalize numeric join-key fields so 1, 1.0, and '1.0' match."""

    if _is_missing_scalar(value):
        return ""
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not pd.isna(parsed):
        number = float(parsed)
        if math.isfinite(number):
            return str(int(number)) if number.is_integer() else f"{number:g}"
    return str(value).strip()


def _protocol3_group_key(row: pd.Series) -> tuple[str, str, str, str]:
    k_value = _first_present(row, ("k_per_class", "target_calibration_per_class"))
    return (
        _canonical_text_key(row.get("method", "")),
        _canonical_text_key(row.get("outer_test_subject", "")),
        _canonical_numeric_key(row.get("fold_index", "")),
        _canonical_numeric_key(k_value),
    )


def install() -> None:
    from neureptrace import bushmeg_all_protocols_audit

    bushmeg_all_protocols_audit._parse_index_set = _parse_index_set
    bushmeg_all_protocols_audit._protocol3_group_key = _protocol3_group_key
