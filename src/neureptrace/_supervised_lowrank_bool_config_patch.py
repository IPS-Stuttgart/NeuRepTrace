"""Runtime guardrail for supervised-lowrank boolean config parsing."""

from __future__ import annotations

from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_supervised_lowrank_bool_config_patch_installed"


def _as_bool(value: Any) -> bool:
    """Normalize only unambiguous supervised-lowrank boolean tokens."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        if int(value) in {0, 1}:
            return bool(value)
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def install() -> None:
    """Install strict boolean parsing for supervised-lowrank config flags."""

    from neureptrace import bushmeg_supervised_lowrank_loso

    if getattr(bushmeg_supervised_lowrank_loso, _PATCH_MARKER, False):
        return

    bushmeg_supervised_lowrank_loso._as_bool = _as_bool
    setattr(bushmeg_supervised_lowrank_loso, _PATCH_MARKER, True)


__all__ = ["install"]
