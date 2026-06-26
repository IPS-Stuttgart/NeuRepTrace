"""Runtime patch for safe confusion-metadata label lookup."""

from __future__ import annotations

import numpy as np


def _integer_like_metadata_key(label: object) -> int | None:
    if isinstance(label, (bool, np.bool_)):
        return None
    if isinstance(label, (int, np.integer)):
        return int(label)
    if isinstance(label, str):
        try:
            return int(label)
        except (TypeError, ValueError, OverflowError):
            return None
    try:
        numeric = float(label)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def install() -> None:
    """Install a metadata lookup guard that never truncates fractional labels."""
    import neureptrace.metrics.confusion as confusion_metrics

    if getattr(confusion_metrics._lookup_label_metadata, "_fractional_metadata_lookup_patched", False):
        return

    def _lookup_label_metadata(metadata_by_label: dict[object, dict[str, object]], label: object) -> dict[str, object]:
        if label in metadata_by_label:
            return metadata_by_label[label]
        label_text = str(label)
        if label_text in metadata_by_label:
            return metadata_by_label[label_text]
        label_int = _integer_like_metadata_key(label)
        if label_int is None:
            return {}
        return metadata_by_label.get(label_int, {})

    _lookup_label_metadata._fractional_metadata_lookup_patched = True  # type: ignore[attr-defined]
    confusion_metrics._lookup_label_metadata = _lookup_label_metadata


__all__ = ["install"]
