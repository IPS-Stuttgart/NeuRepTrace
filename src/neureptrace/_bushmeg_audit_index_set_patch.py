"""Patch list-like calibration-index parsing in the BUSH-MEG audit."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


def _is_missing_scalar(value: Any) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, bool):
        return missing
    return False


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


def install() -> None:
    from neureptrace import bushmeg_all_protocols_audit

    bushmeg_all_protocols_audit._parse_index_set = _parse_index_set
