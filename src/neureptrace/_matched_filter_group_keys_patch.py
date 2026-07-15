"""Preserve exact and missing matched-filter group and stream identifiers."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from neureptrace._object_label_utils import label_equal_mask

_PATCH_MARKER = "_neureptrace_matched_filter_group_keys_patch_installed"


def _grouped(frame: pd.DataFrame, columns: Sequence[str], *, sort: bool = True):
    """Group rows without discarding missing-valued identifiers."""

    if not columns:
        return [((), frame)]
    by: str | list[str] = columns[0] if len(columns) == 1 else list(columns)
    return frame.groupby(by, sort=sort, dropna=False)


def _key_values(key: object, columns: Sequence[str]) -> dict[str, object]:
    """Map group keys without unpacking tuple-valued single-column identifiers."""

    values = (key,) if len(columns) == 1 else key if isinstance(key, tuple) else (key,)
    return dict(zip(columns, values, strict=True))


def _filter_by_values(frame: pd.DataFrame, values: dict[str, object]) -> pd.DataFrame:
    """Filter identifiers using NeuRepTrace's exact, missing-aware equality."""

    filtered = frame
    for column, value in values.items():
        if column in filtered.columns:
            mask = label_equal_mask(filtered[column].to_numpy(dtype=object), value)
            filtered = filtered.loc[mask]
    return filtered


def install() -> None:
    """Install robust group-key handling on the matched-filter module."""

    from neureptrace import matched_filter_detection

    if getattr(matched_filter_detection, _PATCH_MARKER, False):
        return
    matched_filter_detection._grouped = _grouped  # noqa: SLF001
    matched_filter_detection._key_values = _key_values  # noqa: SLF001
    matched_filter_detection._filter_by_values = _filter_by_values  # noqa: SLF001
    setattr(matched_filter_detection, _PATCH_MARKER, True)


__all__ = ["install"]
