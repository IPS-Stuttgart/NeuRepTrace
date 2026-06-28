"""Allow structured OpenNeuro alignment manifest fallback values."""

from __future__ import annotations

import importlib
from typing import Any

_PATCH_MARKER = "_neureptrace_openneuro_alignment_compare_structured_values_patch_installed"


def _first_nonempty(*values: Any) -> str:
    """Return the first non-empty value as text without hashing structured values."""

    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def install() -> None:
    """Patch OpenNeuro alignment comparison structured-value handling."""

    module = importlib.import_module("neureptrace.openneuro_alignment_compare")
    if getattr(module, _PATCH_MARKER, False):
        return
    module._first_nonempty = _first_nonempty
    setattr(module, _PATCH_MARKER, True)


__all__ = ["install"]
