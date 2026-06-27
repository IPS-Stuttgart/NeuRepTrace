"""Normalize boolean-like OpenNeuro manifest compatibility tokens."""

from __future__ import annotations

import importlib
from typing import Any

_PATCH_MARKER = "_neureptrace_openneuro_manifest_bool_token_patch_installed"

BOOLEAN_MANIFEST_COMPATIBILITY_COLUMNS = {
    "run_decode",
    "skip_failed_subjects",
    "temporal_smoothing",
    "response_window_ensemble",
    "ensemble_source_baseline_debiasing",
    "label_shuffle_control",
}

_TRUE_TOKENS = {"1", "true", "yes", "y", "on"}
_FALSE_TOKENS = {"0", "false", "no", "n", "off"}


def _bool_token(value: Any) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in _TRUE_TOKENS:
        return "true"
    if text in _FALSE_TOKENS:
        return "false"
    return None


def install() -> None:
    """Normalize bool-like manifest tokens before cross-shard compatibility checks."""

    module = importlib.import_module("neureptrace.openneuro_decode_diagnostics")
    original = module._manifest_compatibility_token
    if getattr(original, _PATCH_MARKER, False):
        return

    def _manifest_compatibility_token(column: str, value: Any) -> str:
        if column in BOOLEAN_MANIFEST_COMPATIBILITY_COLUMNS:
            token = _bool_token(value)
            if token is not None:
                return token
        return original(column, value)

    setattr(_manifest_compatibility_token, _PATCH_MARKER, True)
    module._manifest_compatibility_token = _manifest_compatibility_token


__all__ = ["install"]
