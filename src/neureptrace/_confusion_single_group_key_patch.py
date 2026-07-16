"""Harden confusion grouping and required-column validation.

Pandas 2.x returns a one-element tuple when grouping by a one-column list. The
legacy confusion helper wrapped that tuple again, so public grouped summaries
exposed values such as ``("logistic",)`` instead of ``"logistic"``. Grouping
by the scalar column name gives a stable scalar key while still preserving
tuple-valued identifiers as atomic group values.

Confusion helpers also assign semantic roles to true-label, predicted-label,
group, and participant columns. Reusing one physical column for multiple roles,
or accepting a frame with duplicate required column names, makes pandas select
or rename ambiguous columns and fail later with opaque errors. The shared guard
now rejects those inputs before any summary is computed.
"""

from __future__ import annotations

import importlib
from collections import Counter
from collections.abc import Sequence

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


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Reject missing, role-colliding, or physically duplicated columns."""

    requested = list(columns)
    repeated_roles = sorted(column for column, count in Counter(requested).items() if count > 1)
    if repeated_roles:
        raise ValueError(
            "Confusion column roles must reference distinct columns; "
            f"repeated columns: {repeated_roles}"
        )

    missing = [column for column in requested if column not in frame.columns]
    if missing:
        raise ValueError(f"Data frame is missing required columns: {missing}")

    frame_columns = frame.columns.tolist()
    ambiguous = [column for column in requested if frame_columns.count(column) > 1]
    if ambiguous:
        raise ValueError(f"Data frame has ambiguous duplicate required columns: {ambiguous}")


def install() -> None:
    """Install stable grouping and deterministic column validation."""

    module = importlib.import_module("neureptrace.metrics.confusion")
    if getattr(module, _PATCH_MARKER, False):
        return
    module._iter_frame_groups = _iter_frame_groups
    module._require_columns = _require_columns
    setattr(module, _PATCH_MARKER, True)
