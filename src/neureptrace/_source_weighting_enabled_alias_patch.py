"""Runtime patches for source-group weighting mode aliases and score validation."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_AFFIRMATIVE_ENABLE_ALIASES = {"1", "true", "on", "yes", "enable", "enabled"}
_DISABLED_ENABLE_ALIASES = {"0", "false", "off", "no", "disable", "disabled"}
_MODE_ALIAS_PATCH_ATTR = "_enabled_alias_patch"
_SCORE_BOOL_PATCH_ATTR = "_boolean_score_patch"


def _normalized_alias(value: Any) -> str:
    return "none" if value is None else str(value).strip().lower().replace("-", "_")


def _is_boolean_score(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return isinstance(value.item(), (bool, np.bool_))
        return np.issubdtype(value.dtype, np.bool_)
    return False


def install() -> None:
    """Install boolean-like aliases and numeric score guardrails."""
    import neureptrace.decoding.source_weighting as source_weighting

    original_normalize_source_group_weighting_mode = source_weighting.normalize_source_group_weighting_mode
    if not getattr(original_normalize_source_group_weighting_mode, _MODE_ALIAS_PATCH_ATTR, False):

        @wraps(original_normalize_source_group_weighting_mode)
        def normalize_source_group_weighting_mode(value: Any) -> str:
            normalized = _normalized_alias(value)
            if normalized in _AFFIRMATIVE_ENABLE_ALIASES:
                return "source_reliability"
            if normalized in _DISABLED_ENABLE_ALIASES:
                return "none"
            return original_normalize_source_group_weighting_mode(value)

        setattr(normalize_source_group_weighting_mode, _MODE_ALIAS_PATCH_ATTR, True)
        source_weighting.normalize_source_group_weighting_mode = normalize_source_group_weighting_mode

    original_score_to_utility = source_weighting._score_to_utility
    if not getattr(original_score_to_utility, _SCORE_BOOL_PATCH_ATTR, False):

        @wraps(original_score_to_utility)
        def _score_to_utility(score: Any, *, metric: str) -> float:
            if _is_boolean_score(score):
                raise ValueError("source-group scores must be numeric values, not booleans.")
            return original_score_to_utility(score, metric=metric)

        setattr(_score_to_utility, _SCORE_BOOL_PATCH_ATTR, True)
        source_weighting._score_to_utility = _score_to_utility


__all__ = ["install"]
