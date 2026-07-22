"""Handle composite labels and class-count metadata in BUSH-MEG summaries."""

from __future__ import annotations

import importlib
import math
from collections import Counter
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_bushmeg_profile_label_counts_patch_installed"
_AUDIT_PATCH_MARKER = "_neureptrace_bushmeg_audit_count_validation_patch_installed"


def _normalise_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _normalise_value(value.item())
        return tuple(_normalise_value(item) for item in value.tolist())
    if isinstance(value, list):
        return tuple(_normalise_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_normalise_value(item) for item in value)
    return value


def _label_values(labels: Any) -> list[Any]:
    array = np.asarray(labels, dtype=object)
    if array.ndim == 0:
        return [_normalise_value(array.item())]
    if array.ndim == 1:
        return [_normalise_value(value) for value in array.tolist()]
    if array.ndim == 2 and array.shape[1] == 1:
        return [_normalise_value(value) for value in array[:, 0].tolist()]
    return [_normalise_value(tuple(row)) for row in array.reshape(array.shape[0], -1).tolist()]


def _count_key(value: Any) -> str:
    value = _normalise_value(value)
    return repr(value) if isinstance(value, tuple) else str(value)


def _class_count_dict(labels: Any) -> dict[str, int]:
    counts = Counter(_count_key(value) for value in _label_values(labels))
    return {key: int(counts[key]) for key in sorted(counts)}


def _contains_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return True
        return any(_contains_boolean(item) for item in value.reshape(-1).tolist())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_boolean(item) for item in value)
    return False


def _integer_count_or_none(audit: Any, value: Any, *, minimum: int) -> int | None:
    if _contains_boolean(value):
        return None
    numeric = audit._numeric_or_na(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    count = int(numeric)
    return count if count >= minimum else None


def _protocol3_calibration_count_failures(summary: Any) -> list[str]:
    audit = importlib.import_module("neureptrace.bushmeg_all_protocols_audit")
    p3 = audit._protocol_rows(summary, 3)
    if p3.empty:
        return []
    if "n_target_calibration_trials" not in p3.columns:
        return ["Protocol 3 summary rows are missing required column `n_target_calibration_trials`."]

    failures: list[str] = []
    for _, row in p3.iterrows():
        if audit._row_is_explicitly_skipped(row):
            continue

        k_raw = audit._first_present(row, ("k_per_class", "target_calibration_per_class"))
        n_classes_raw = audit._class_count_from_row(row)
        n_calibration_raw = row["n_target_calibration_trials"]
        k_value = _integer_count_or_none(audit, k_raw, minimum=0)
        n_classes = (
            None
            if "n_classes" in row.index and _contains_boolean(row["n_classes"])
            else _integer_count_or_none(audit, n_classes_raw, minimum=1)
        )
        n_calibration = _integer_count_or_none(audit, n_calibration_raw, minimum=0)

        if k_value is None or n_classes is None or n_calibration is None:
            failures.append(
                "Protocol 3 calibration counts must be finite integers with k >= 0, "
                "n_classes >= 1, and n_target_calibration_trials >= 0 for "
                f"{audit._row_label(row)}."
            )
            continue

        expected = k_value * n_classes
        if n_calibration != expected:
            failures.append(
                f"Protocol 3 calibration count mismatch for {audit._row_label(row)}: "
                f"n_target_calibration_trials={n_calibration} but k*n_classes={expected}."
            )
    return failures


def install() -> None:
    importlib.import_module("neureptrace._bushmeg_diagnostics_class_count_patch").install()

    audit = importlib.import_module("neureptrace.bushmeg_all_protocols_audit")
    if not getattr(audit, _AUDIT_PATCH_MARKER, False):
        audit._protocol3_calibration_count_failures = _protocol3_calibration_count_failures
        setattr(audit, _AUDIT_PATCH_MARKER, True)

    all_protocols = importlib.import_module("neureptrace.bushmeg_all_protocols")
    if getattr(all_protocols, _PATCH_MARKER, False):
        return
    all_protocols._class_count_dict = _class_count_dict
    setattr(all_protocols, _PATCH_MARKER, True)


__all__ = ["install"]
