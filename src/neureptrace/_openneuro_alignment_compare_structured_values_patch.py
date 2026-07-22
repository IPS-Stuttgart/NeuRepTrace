"""Guard structured and non-finite OpenNeuro alignment comparison inputs."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

_PATCH_MARKER = "_neureptrace_openneuro_alignment_compare_structured_values_patch_installed"
_FINITE_METRIC_PATCH_MARKER = "_neureptrace_openneuro_alignment_compare_finite_metric_patch_installed"

_TRUE_TOKENS = {"1", "true", "yes", "y", "on"}
_FALSE_TOKENS = {"0", "false", "no", "n", "off"}


def _is_missing_scalar(value: Any) -> bool:
    """Return true for scalar pandas/numpy missing values only."""

    try:
        missing = np.asarray(pd.isna(value))
    except (TypeError, ValueError):
        return False
    if missing.ndim != 0:
        return False
    return bool(missing.item())


def _first_nonempty(*values: Any) -> str:
    """Return the first non-empty, non-missing value as text."""

    for value in values:
        if value is None or _is_missing_scalar(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _bool_tokens(value: Any) -> list[str]:
    """Normalize scalar or sequence boolean-like values into lowercase tokens."""

    if value is None or _is_missing_scalar(value):
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(token).strip().lower() for token in value if str(token).strip()]
    return [str(value).strip().lower()] if str(value).strip() else []


def _as_bool(value: Any) -> bool:
    """Parse scalar or sequence manifest/CSV boolean values without ambiguous pd.isna checks."""

    if isinstance(value, bool):
        return value
    tokens = _bool_tokens(value)
    if not tokens:
        return False
    parsed: set[bool] = set()
    for token in tokens:
        if token in _TRUE_TOKENS:
            parsed.add(True)
        elif token in _FALSE_TOKENS:
            parsed.add(False)
        else:
            parsed.add(False)
    if len(parsed) > 1:
        raise ValueError(f"Inconsistent boolean provenance: {tokens}")
    return parsed.pop()


def _install_finite_metric_selection(module: Any) -> None:
    """Exclude non-finite time/metric rows before ranking alignment variants."""

    if getattr(module, _FINITE_METRIC_PATCH_MARKER, False):
        return

    original_select_metric = module._select_metric

    @wraps(original_select_metric)
    def select_metric(
        summary: pd.DataFrame,
        *,
        metric: str,
        fixed_time: float | None,
    ) -> dict[str, Any]:
        fixed_time_value = None if fixed_time is None else float(fixed_time)
        if fixed_time_value is not None and not np.isfinite(fixed_time_value):
            raise ValueError("fixed_time must be finite.")

        if "time" in summary.columns and metric in summary.columns:
            times = pd.to_numeric(summary["time"], errors="coerce").to_numpy(dtype=float)
            scores = pd.to_numeric(summary[metric], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(times) & np.isfinite(scores)
            positions = np.flatnonzero(finite)
            summary = summary.iloc[positions].copy()
            summary["time"] = times[positions]
            summary[metric] = scores[positions]

        return original_select_metric(summary, metric=metric, fixed_time=fixed_time_value)

    module._select_metric = select_metric
    setattr(module, _FINITE_METRIC_PATCH_MARKER, True)


def install() -> None:
    """Patch OpenNeuro alignment comparison input handling."""

    module = importlib.import_module("neureptrace.openneuro_alignment_compare")
    if not getattr(module, _PATCH_MARKER, False):
        module._first_nonempty = _first_nonempty
        module._as_bool = _as_bool
        setattr(module, _PATCH_MARKER, True)
    _install_finite_metric_selection(module)


__all__ = ["install"]
