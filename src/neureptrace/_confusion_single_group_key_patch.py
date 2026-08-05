"""Harden confusion grouping, label-pair ordering, and column validation.

Pandas 2.x returns a one-element tuple when grouping by a one-column list. The
legacy confusion helper wrapped that tuple again, so public grouped summaries
exposed values such as ``("logistic",)`` instead of ``"logistic"``. Grouping
by the scalar column name gives a stable scalar key while still preserving
tuple-valued identifiers as atomic group values.

Confusion-pair summaries also canonicalize unordered label pairs through a sort
key. Numeric labels and numeric-looking string labels can share the same legacy
sort key, for example integer ``1`` and string ``"1"``. Python's stable sort
then preserves the directional input order, splitting opposite-direction errors
across two pair rows. A deterministic type-and-representation tie breaker keeps
such distinct labels in one canonical unordered pair.

Confusion helpers also assign semantic roles to true-label, predicted-label,
group, and participant columns. Reusing one physical column for multiple roles,
or accepting a frame with duplicate required column names, makes pandas select
or rename ambiguous columns and fail later with opaque errors. The shared guard
rejects those inputs before any summary is computed.

Finally, confusion summaries generate canonical label and metric columns.
Distinct source columns can still collide with those generated names after
renaming, or a generated metric can overwrite a group identifier in the output
row. Public wrappers reject those schema collisions before pandas grouping or
result construction.
"""

from __future__ import annotations

import importlib
from collections import Counter
from collections.abc import Sequence
from functools import wraps

import pandas as pd

_PATCH_MARKER = "_neureptrace_confusion_single_group_key_patch_installed"
_CONFUSION_COUNT_OUTPUT_COLUMNS = frozenset({"true_label", "predicted_label", "count"})
_PER_CLASS_ACCURACY_OUTPUT_COLUMNS = frozenset(
    {"true_label", "n_trials", "n_correct", "accuracy", "n_participants"}
)


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


def _label_sort_key(value: object) -> tuple[int, float | str, str, str]:
    """Return a direction-independent total-order key for scalar labels."""

    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    try:
        return 0, float(value), type_name, repr(value)
    except (TypeError, ValueError):
        return 1, str(value), type_name, repr(value)


def _ordered_label_pair(first: object, second: object) -> tuple[object, object]:
    """Canonicalize an unordered label pair even when primary sort keys tie."""

    return tuple(sorted((first, second), key=_label_sort_key))


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


def _validate_generated_column_collisions(
    group_columns: Sequence[str],
    output_columns: frozenset[str],
    *,
    operation: str,
) -> None:
    collisions = sorted(set(group_columns).intersection(output_columns))
    if collisions:
        raise ValueError(
            f"group_columns overlap generated {operation} columns: {collisions}"
        )


def _validate_participant_label_alias(
    participant_column: str | None,
    *,
    true_column: str,
    predicted_column: str,
) -> None:
    if participant_column is None:
        return

    generated_aliases: set[str] = set()
    if true_column != "true_label":
        generated_aliases.add("true_label")
    if predicted_column != "predicted_label":
        generated_aliases.add("predicted_label")
    if participant_column in generated_aliases:
        raise ValueError(
            "participant_column conflicts with a generated per-class label column: "
            f"{participant_column!r}"
        )


def install() -> None:
    """Install stable grouping, pair ordering, and column validation."""

    module = importlib.import_module("neureptrace.metrics.confusion")
    if getattr(module, _PATCH_MARKER, False):
        return

    public_metrics = importlib.import_module("neureptrace.metrics")
    original_confusion_counts = module.confusion_counts
    original_per_class_accuracy = module.per_class_accuracy

    @wraps(original_confusion_counts)
    def confusion_counts(
        frame: pd.DataFrame,
        true_column: str = "true_label",
        predicted_column: str = "predicted_label",
        group_columns: Sequence[str] = (),
    ) -> pd.DataFrame:
        normalized_groups = module._normalize_columns(group_columns)
        _validate_generated_column_collisions(
            normalized_groups,
            _CONFUSION_COUNT_OUTPUT_COLUMNS,
            operation="confusion-count",
        )
        return original_confusion_counts(
            frame,
            true_column=true_column,
            predicted_column=predicted_column,
            group_columns=normalized_groups,
        )

    @wraps(original_per_class_accuracy)
    def per_class_accuracy(
        frame: pd.DataFrame,
        true_column: str = "true_label",
        predicted_column: str = "predicted_label",
        participant_column: str | None = None,
        group_columns: Sequence[str] = (),
    ) -> pd.DataFrame:
        normalized_groups = module._normalize_columns(group_columns)
        _validate_generated_column_collisions(
            normalized_groups,
            _PER_CLASS_ACCURACY_OUTPUT_COLUMNS,
            operation="per-class-accuracy",
        )
        _validate_participant_label_alias(
            participant_column,
            true_column=true_column,
            predicted_column=predicted_column,
        )
        return original_per_class_accuracy(
            frame,
            true_column=true_column,
            predicted_column=predicted_column,
            participant_column=participant_column,
            group_columns=normalized_groups,
        )

    module._iter_frame_groups = _iter_frame_groups
    module._label_sort_key = _label_sort_key
    module._ordered_label_pair = _ordered_label_pair
    module._require_columns = _require_columns
    module.confusion_counts = confusion_counts
    module.per_class_accuracy = per_class_accuracy
    public_metrics.confusion_counts = confusion_counts
    public_metrics.per_class_accuracy = per_class_accuracy
    setattr(module, _PATCH_MARKER, True)
