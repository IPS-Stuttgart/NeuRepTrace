"""Runtime patch for strict top-level FieldTrip label configuration."""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_fieldtrip_label_config_validation_patched"
_LABEL_BASE_ERROR = "label_base must be a finite numeric scalar or None, not a boolean value."
_LABEL_BASE_PARSE_ERROR = "label-base must be finite numeric or 'none', not a boolean value."
_TRIALINFO_COLUMN_ERROR = "trialinfo_column must be a finite integer column index, not a boolean value."


def _scalar_numeric_value(value: Any, *, message: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    return value


def _coerce_label_base(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "none", "null"}:
            return None
        value = text
    value = _scalar_numeric_value(value, message=_LABEL_BASE_ERROR)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(_LABEL_BASE_ERROR) from exc
    if not math.isfinite(parsed):
        raise ValueError(_LABEL_BASE_ERROR)
    return parsed


def _coerce_trialinfo_column(value: Any) -> int:
    value = _scalar_numeric_value(value, message=_TRIALINFO_COLUMN_ERROR)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(_TRIALINFO_COLUMN_ERROR) from exc
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ValueError(_TRIALINFO_COLUMN_ERROR)
    return int(parsed)


def install() -> None:
    """Install validation for top-level FieldTrip label config helpers."""

    import neureptrace.fieldtrip_mat as fieldtrip_mat

    if not getattr(fieldtrip_mat._parse_label_base, _PATCH_MARKER, False):
        original_parse_label_base = fieldtrip_mat._parse_label_base

        def _parse_label_base(value: Any) -> float | None:
            try:
                return _coerce_label_base(value)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(_LABEL_BASE_PARSE_ERROR) from exc

        _parse_label_base.__wrapped__ = original_parse_label_base
        setattr(_parse_label_base, _PATCH_MARKER, True)
        fieldtrip_mat._parse_label_base = _parse_label_base

    if getattr(fieldtrip_mat._metadata_from_trialinfo, _PATCH_MARKER, False):
        return

    original_metadata_from_trialinfo = fieldtrip_mat._metadata_from_trialinfo

    def _metadata_from_trialinfo(
        *,
        n_trials: int,
        trialinfo: np.ndarray | None,
        sampleinfo: np.ndarray | None,
        label_column: str,
        label_base: Any,
        trialinfo_column: Any,
    ) -> Any:
        return original_metadata_from_trialinfo(
            n_trials=n_trials,
            trialinfo=trialinfo,
            sampleinfo=sampleinfo,
            label_column=label_column,
            label_base=_coerce_label_base(label_base),
            trialinfo_column=_coerce_trialinfo_column(trialinfo_column),
        )

    _metadata_from_trialinfo.__wrapped__ = original_metadata_from_trialinfo
    setattr(_metadata_from_trialinfo, _PATCH_MARKER, True)
    fieldtrip_mat._metadata_from_trialinfo = _metadata_from_trialinfo
