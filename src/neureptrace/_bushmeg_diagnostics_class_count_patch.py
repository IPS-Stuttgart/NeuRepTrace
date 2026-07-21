"""Validate class-count metadata before inferring BUSH-MEG chance levels."""

from __future__ import annotations

import importlib
from functools import wraps

import numpy as np
import pandas as pd

_PATCH_MARKER = "_neureptrace_bushmeg_diagnostics_class_count_patch_installed"


def _uniform_integral_class_count(summary: pd.DataFrame) -> int | None:
    """Return one trustworthy class count, tolerating only missing entries."""

    if "n_classes" not in summary.columns:
        return None
    raw = summary["n_classes"]
    present = raw.notna()
    if not present.any():
        return None
    numeric = pd.to_numeric(raw, errors="coerce")
    populated = numeric[present]
    if populated.isna().any() or not np.all(np.isfinite(populated)):
        return None
    values = populated.to_numpy(dtype=float)
    if not np.all(values > 1.0) or not np.all(values == np.floor(values)):
        return None
    unique = np.unique(values)
    if unique.size != 1:
        return None
    return int(unique[0])


def install() -> None:
    """Prevent invalid class counts from being used during chance inference."""

    diagnostics = importlib.import_module("neureptrace.bushmeg_diagnostics")
    original = diagnostics.infer_balanced_accuracy_chance
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def infer_balanced_accuracy_chance(
        summary: pd.DataFrame,
        predictions: pd.DataFrame | None = None,
        *,
        chance: float | None = None,
    ) -> float:
        if chance is not None:
            return original(summary, predictions, chance=chance)

        n_classes = _uniform_integral_class_count(summary)
        if n_classes is not None:
            return 1.0 / float(n_classes)

        if "n_classes" in summary.columns:
            summary = summary.drop(columns=["n_classes"])
        return original(summary, predictions, chance=None)

    setattr(infer_balanced_accuracy_chance, _PATCH_MARKER, True)
    diagnostics.infer_balanced_accuracy_chance = infer_balanced_accuracy_chance


__all__ = ["install"]
