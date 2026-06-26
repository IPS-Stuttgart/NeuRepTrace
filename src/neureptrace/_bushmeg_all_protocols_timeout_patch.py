"""Validate BUSH-MEG all-protocol timeout controls strictly."""

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
    return isinstance(value, (bool, np.bool_))


def _validate_timeout_seconds(name: str, value: float | int | None) -> float | None:
    if value is None:
        return None
    if _is_bool_scalar(value):
        raise ValueError(f"{name} must be a positive finite number when provided.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number when provided.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive finite number when provided.")
    return parsed


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
    """Patch BUSH-MEG all-protocol timeout validation and timeout status writes."""

    all_protocols = importlib.import_module("neureptrace.bushmeg_all_protocols")
    if getattr(all_protocols, _PATCH_MARKER, False):
        return

    all_protocols._validate_timeout_seconds = _validate_timeout_seconds
    _patch_pre_runner_status_timeout(all_protocols)
    setattr(all_protocols, _PATCH_MARKER, True)


__all__ = ["install"]
