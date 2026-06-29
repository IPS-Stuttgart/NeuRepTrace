"""Runtime patch for safe confusion-metadata label lookup."""

from __future__ import annotations

import numpy as np

_PATCH_MARKER = "_neureptrace_confusion_metadata_lookup_patch_installed"
_MISSING = object()


def _canonical_label(value: object) -> object:
    """Return a hashable label key while preserving scalar label semantics."""

    if isinstance(value, np.generic):
        return _canonical_label(value.item())
    if isinstance(value, np.ndarray):
        array = value.astype(object, copy=False)
        if array.ndim == 0:
            return _canonical_label(array.item())
        flat = array.reshape(-1)
        if flat.size == 1:
            return _canonical_label(flat[0])
        return tuple(_canonical_label(item) for item in flat.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_label(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return str(value)
    return value


def _metadata_lookup(metadata_by_label: dict[object, dict[str, object]], label: object) -> dict[str, object] | object:
    key = _canonical_label(label)
    try:
        if key in metadata_by_label:
            return metadata_by_label[key]
    except TypeError:
        return _MISSING
    return _MISSING


def _integer_like_metadata_key(label: object) -> int | None:
    label = _canonical_label(label)
    if isinstance(label, (bool, np.bool_, tuple)):
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
    """Install metadata lookup guards for fractional and composite labels."""
    import neureptrace.metrics.confusion as confusion_metrics

    if getattr(confusion_metrics, _PATCH_MARKER, False):
        return

    original_prediction_frame = confusion_metrics._prediction_frame
    original_metadata_label_id = confusion_metrics._metadata_label_id

    def _prediction_frame(frame, *, true_column: str, predicted_column: str, group_columns, participant_column: str | None):
        working = original_prediction_frame(
            frame,
            true_column=true_column,
            predicted_column=predicted_column,
            group_columns=group_columns,
            participant_column=participant_column,
        )
        for column in (confusion_metrics._TRUE_LABEL, confusion_metrics._PREDICTED_LABEL):
            working[column] = [_canonical_label(value) for value in working[column].tolist()]
        return working

    def _metadata_by_label(metadata_frame, metadata_label_columns) -> dict[object, dict[str, object]]:
        metadata_by_label: dict[object, dict[str, object]] = {}
        if metadata_frame is None or metadata_frame.empty:
            return metadata_by_label
        label_columns = confusion_metrics._normalize_columns(metadata_label_columns)
        for _, metadata_row in metadata_frame.iterrows():
            label = original_metadata_label_id(metadata_row, label_columns)
            if label is not None:
                metadata_by_label[_canonical_label(label)] = metadata_row.to_dict()
        return metadata_by_label

    def _lookup_label_metadata(metadata_by_label: dict[object, dict[str, object]], label: object) -> dict[str, object]:
        label_key = _canonical_label(label)
        direct_match = _metadata_lookup(metadata_by_label, label_key)
        if direct_match is not _MISSING:
            return direct_match
        label_text = str(label_key)
        text_match = _metadata_lookup(metadata_by_label, label_text)
        if text_match is not _MISSING:
            return text_match
        label_int = _integer_like_metadata_key(label_key)
        if label_int is None:
            return {}
        int_match = _metadata_lookup(metadata_by_label, label_int)
        if int_match is not _MISSING:
            return int_match
        return {}

    confusion_metrics._prediction_frame = _prediction_frame
    confusion_metrics._metadata_by_label = _metadata_by_label
    confusion_metrics._lookup_label_metadata = _lookup_label_metadata
    setattr(confusion_metrics, _PATCH_MARKER, True)


__all__ = ["install"]
