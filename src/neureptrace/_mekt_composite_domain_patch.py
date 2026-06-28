"""Patch MEKT composite domain handling for homogeneous matrix labels.

NumPy turns homogeneous lists of tuple/list domain identifiers into regular
2-D arrays.  The MEKT helper path used to flatten those arrays, which destroyed
row-wise composite domain IDs before DTE selection and per-domain alignment.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _row_tuple_list(array: np.ndarray) -> list[tuple[Any, ...]]:
    rows = np.asarray(array, dtype=object).reshape(array.shape[0], -1)
    return [tuple(row.tolist()) for row in rows]


def _patched_value_list(values: Sequence[Any] | np.ndarray) -> list[Any]:
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            return [array.item()]
        if array.ndim == 1:
            return array.tolist()
        return _row_tuple_list(array)
    if isinstance(values, (str, bytes)):
        return [values]
    try:
        return list(values)
    except TypeError:
        return [values]


def _patched_domain_ids(n_rows: int, source_domains: Sequence[Any] | np.ndarray | None, *, name: str) -> np.ndarray:
    if source_domains is None:
        return np.zeros(n_rows, dtype=int)
    raw = np.asarray(source_domains)
    if raw.shape[0] != n_rows:
        raise ValueError(f"{name} length must match source rows.")
    if raw.ndim >= 2:
        return np.asarray(_row_tuple_list(raw), dtype=object)
    return raw


def install() -> None:
    from neureptrace.decoding import mekt

    mekt._value_list = _patched_value_list
    mekt._domain_ids = _patched_domain_ids
