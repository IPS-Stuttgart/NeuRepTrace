"""Accept a single string group column in temporal-generalization summaries."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

_PATCH_MARKER = "_neureptrace_temporal_generalization_string_groups_patch_installed"


def _normalize_group_columns(group_columns: Sequence[str] | str | None) -> list[str]:
    if group_columns is None:
        return []
    if isinstance(group_columns, str):
        return [group_columns]
    return list(dict.fromkeys(group_columns))


def install() -> None:
    """Patch summary grouping so ``group_columns=\"decoder\"`` means one column."""

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
            frame,
            group_columns=_normalize_group_columns(group_columns),
            accuracy_column=accuracy_column,
            chance_column=chance_column,
        )

    setattr(summarize_temporal_generalization_matrix, _PATCH_MARKER, True)
    temporal_generalization.summarize_temporal_generalization_matrix = summarize_temporal_generalization_matrix


__all__ = ["install"]
