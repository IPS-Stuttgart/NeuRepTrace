"""Reject inconsistent temporal-model probability-class metadata."""

from __future__ import annotations

import importlib

import pandas as pd

_PATCH_MARKER = "_neureptrace_temporal_model_class_metadata_patch_installed"
_MISSING_CLASS_TOKENS = {"", "nan", "none", "nat"}


def _class_values(frame: pd.DataFrame, class_column: str) -> list[str]:
    """Return distinct, normalized non-missing class labels in row order."""

    values = frame[class_column]
    normalized = values.loc[values.notna()].astype(str).str.strip()
    normalized = normalized.loc[~normalized.str.lower().isin(_MISSING_CLASS_TOKENS)]
    return normalized.drop_duplicates().tolist()


def _class_names(frame: pd.DataFrame, prob_columns: list[str]) -> list[str]:
    """Resolve one unambiguous class name for every probability column."""

    names: list[str] = []
    for index, probability_column in enumerate(prob_columns):
        suffix = probability_column.removeprefix("prob_class_")
        class_column = f"class_{suffix}"
        values = _class_values(frame, class_column) if class_column in frame.columns else []
        if len(values) > 1:
            raise ValueError(
                "Temporal-model class metadata must be consistent within each decoder/emission group; "
                f"{class_column} maps to multiple classes: {values[:5]}."
            )
        names.append(values[0] if values else (suffix if suffix.isdigit() else str(index)))

    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(
            "Temporal-model class metadata must map probability columns to distinct classes; "
            f"duplicate class names: {duplicate_names[:5]}."
        )
    return names


def install() -> None:
    """Install strict class metadata resolution for temporal models."""

    temporal_model = importlib.import_module("neureptrace.temporal_model")
    if getattr(temporal_model, _PATCH_MARKER, False):
        return

    temporal_model._class_names = _class_names
    setattr(temporal_model, _PATCH_MARKER, True)


__all__ = ["install"]
