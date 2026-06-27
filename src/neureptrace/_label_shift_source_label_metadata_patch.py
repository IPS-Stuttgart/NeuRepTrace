"""Report label-shift source-label usage accurately for explicit-prior runs."""

from __future__ import annotations

import importlib
from functools import wraps

_PATCH_MARKER = "_neureptrace_label_shift_source_label_metadata_patch_installed"


def install() -> None:
    """Patch label-shift adaptation metadata after source-prior-only runs."""

    label_shift = importlib.import_module("neureptrace.decoding.label_shift")
    original_adapt = label_shift.adapt_label_shift_probabilities
    if getattr(original_adapt, _PATCH_MARKER, False):
        return

    @wraps(original_adapt)
    def adapt_label_shift_probabilities(target_probabilities, **kwargs):
        result = original_adapt(target_probabilities, **kwargs)
        uses_source_labels = kwargs.get("source_labels") is not None or kwargs.get("source_validation_labels") is not None
        result.metadata["label_shift_uses_source_labels"] = bool(uses_source_labels)
        return result

    setattr(adapt_label_shift_probabilities, _PATCH_MARKER, True)
    adapt_label_shift_probabilities.__wrapped__ = original_adapt
    label_shift.adapt_label_shift_probabilities = adapt_label_shift_probabilities


__all__ = ["install"]
