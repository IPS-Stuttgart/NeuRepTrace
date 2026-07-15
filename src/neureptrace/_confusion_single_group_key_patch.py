"""Preserve single-column confusion grouping keys across pandas versions.

Pandas 2.x returns a one-element tuple when grouping by a one-column list.
The legacy confusion helper wrapped that tuple again, so public grouped
summaries exposed values such as ``("logistic",)`` instead of ``"logistic"``.
Grouping by the scalar column name gives a stable scalar key while still
preserving tuple-valued identifiers as atomic group values.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import pandas as pd

_PATCH_MARKER = "_neureptrace_confusion_single_group_key_patch_installed"


def _iter_frame_groups(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
):
    if not group_columns:
        yield (), frame
        return

    grouper: str | list[str]
    if len(group_columns) == 1:
        grouper = group_columns[0]
    else:
        grouper = list(group_columns)

    for group_key, group_frame in frame.groupby(grouper, dropna=False, sort=True):
        if len(group_columns) == 1:
            group_key = (group_key,)
        yield tuple(group_key), group_frame


def install() -> None:
    """Install stable single-column grouping for confusion summaries."""

    module = importlib.import_module("neureptrace.metrics.confusion")
    if getattr(module, _PATCH_MARKER, False):
        return
    module._iter_frame_groups = _iter_frame_groups
    setattr(module, _PATCH_MARKER, True)
