"""Runtime patch for affirmative source-group weighting enable aliases."""

from __future__ import annotations

from functools import wraps
from typing import Any

_AFFIRMATIVE_ENABLE_ALIASES = {"true", "on", "yes", "enable", "enabled"}


def _normalized_alias(value: Any) -> str:
    return "none" if value is None else str(value).strip().lower().replace("-", "_")


def install() -> None:
    """Install affirmative boolean aliases for source-group weighting mode parsing."""
    import neureptrace.decoding.source_weighting as source_weighting

    if getattr(source_weighting.normalize_source_group_weighting_mode, "_enabled_alias_patch", False):
        return

    original_normalize_source_group_weighting_mode = source_weighting.normalize_source_group_weighting_mode

    @wraps(original_normalize_source_group_weighting_mode)
    def normalize_source_group_weighting_mode(value: Any) -> str:
        if _normalized_alias(value) in _AFFIRMATIVE_ENABLE_ALIASES:
            return "source_reliability"
        return original_normalize_source_group_weighting_mode(value)

    normalize_source_group_weighting_mode._enabled_alias_patch = True  # type: ignore[attr-defined]
    source_weighting.normalize_source_group_weighting_mode = normalize_source_group_weighting_mode


__all__ = ["install"]
