"""Reject boolean reaction-time CSV trial-index bases."""

from __future__ import annotations

import importlib
from numbers import Integral
from typing import Any

_INSTALLED = False


def _validate_trial_index_base(trial_index_base: Any) -> int:
    module = importlib.import_module("neureptrace.behavior.reaction_time")
    choices = getattr(module, "TRIAL_INDEX_BASE_CHOICES", (0, 1))
    if isinstance(trial_index_base, bool) or not isinstance(trial_index_base, Integral) or trial_index_base not in choices:
        raise ValueError(f"trial_index_base must be one of {choices}, got {trial_index_base!r}.")
    return int(trial_index_base)


def install() -> None:
    """Install the stricter validator into the reaction-time helpers."""

    global _INSTALLED
    if _INSTALLED:
        return
    module = importlib.import_module("neureptrace.behavior.reaction_time")
    module._validate_trial_index_base = _validate_trial_index_base
    _INSTALLED = True
