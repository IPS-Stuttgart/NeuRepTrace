"""Validate BUSH-MEG all-protocol scalar runtime controls strictly."""

from __future__ import annotations

import importlib
import signal
import time
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_bushmeg_all_protocols_timeout_patch_installed"
_ORIGINAL_UPDATE_ATTR = "_neureptrace_original_method_progress_update"
_PRE_RUN_STATUS_TIMEOUT_GUARD_SECONDS = 0.02


def _is_bool_scalar(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray) and value.shape == () and np.issubdtype(value.dtype, np.bool_):
        return True
    return False


def _coerce_numeric_scalar(name: str, value: Any, expectation: str) -> float:
    message = f"{name} must be {expectation}."
    if _is_bool_scalar(value):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
        if _is_bool_scalar(value):
            raise ValueError(message)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(parsed):
        raise ValueError(message)
    return parsed


def _validate_timeout_seconds(name: str, value: float | int | None) -> float | None:
    if value is None:
        return None
    parsed = _coerce_numeric_scalar(name, value, "a positive finite number when provided")
    if parsed <= 0.0:
        raise ValueError(f"{name} must be a positive finite number when provided.")
    return parsed


def _validate_positive_limit(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    parsed = _coerce_numeric_scalar(name, value, "a positive integer")
    if parsed < 1.0 or parsed % 1.0 != 0.0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _signal_timeouts_supported(all_protocols: Any) -> bool:
    checker = getattr(all_protocols, "_signal_timeouts_supported", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except Exception:
        return False


def _patch_pre_runner_status_timeout(all_protocols: Any) -> None:
    progress_cls = all_protocols.MethodProgress
    if hasattr(progress_cls, _ORIGINAL_UPDATE_ATTR):
        return
    original_update = progress_cls.update
    setattr(progress_cls, _ORIGINAL_UPDATE_ATTR, original_update)

    def update(self: Any, stage: str, **fields: Any) -> None:
        method_deadline = getattr(self, "_method_deadline", None)
        if (
            stage != "loading_subjects"
            or method_deadline is None
            or not getattr(self, "_signal_installed", False)
            or not _signal_timeouts_supported(all_protocols)
        ):
            return original_update(self, stage, **fields)

        previous_deadline = float(method_deadline)
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        try:
            self._method_deadline = max(
                previous_deadline,
                time.monotonic() + _PRE_RUN_STATUS_TIMEOUT_GUARD_SECONDS,
            )
            original_update(self, stage, **fields)
        finally:
            self._method_deadline = previous_deadline
            if getattr(self, "_signal_installed", False):
                delay = max(
                    _PRE_RUN_STATUS_TIMEOUT_GUARD_SECONDS,
                    previous_deadline - time.monotonic(),
                )
                signal.setitimer(signal.ITIMER_REAL, delay)

    progress_cls.update = update


def install() -> None:
    """Patch BUSH-MEG all-protocol scalar validation and timeout status writes."""

    all_protocols = importlib.import_module("neureptrace.bushmeg_all_protocols")
    if getattr(all_protocols, _PATCH_MARKER, False):
        return

    all_protocols._validate_timeout_seconds = _validate_timeout_seconds
    all_protocols._validate_positive_limit = _validate_positive_limit
    _patch_pre_runner_status_timeout(all_protocols)
    setattr(all_protocols, _PATCH_MARKER, True)


__all__ = ["install"]
