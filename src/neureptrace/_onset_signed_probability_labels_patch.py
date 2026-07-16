"""Preserve signed integer labels inferred from onset probability columns."""

from __future__ import annotations

from collections.abc import Sequence

_PATCH_MARKER = "_neureptrace_onset_signed_probability_labels_patch_installed"


def _integer_probability_suffix(column: str) -> int | None:
    """Parse a signed decimal label from one ``prob_class_*`` column name."""

    suffix = str(column).removeprefix("prob_class_")
    digits = suffix[1:] if suffix[:1] in {"+", "-"} else suffix
    if not digits or not digits.isdigit():
        return None
    return int(suffix)


def _label_values_from_probability_columns(prob_columns: Sequence[str]) -> tuple[int, ...]:
    labels = tuple(_integer_probability_suffix(column) for column in prob_columns)
    if all(label is not None for label in labels):
        return tuple(int(label) for label in labels if label is not None)
    return tuple(range(len(labels)))


def install() -> None:
    """Install signed-label parsing for onset score and prediction inference."""

    from neureptrace import _event_detection_extensions as extensions

    if getattr(extensions, _PATCH_MARKER, False):
        return
    extensions._label_values_from_probability_columns = _label_values_from_probability_columns
    setattr(extensions, _PATCH_MARKER, True)
