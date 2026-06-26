"""Normalize cue source-weighting top-k disable aliases."""

from __future__ import annotations

from typing import Any

_PATCH_ATTR = "_neureptrace_cue_source_weighting_topk_alias_patch"
_SENTINELS = {"", "none", "null", "off", "all", "full"}


def install() -> None:
    """Install case-insensitive string sentinels for cue ``top_k``."""

    from neureptrace import bushmeg_cue_source_weights

    current = bushmeg_cue_source_weights._normalize_optional_positive_integer
    if getattr(current, _PATCH_ATTR, False):
        return
    original = current

    def _normalize_optional_positive_integer(value: Any, *, name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in _SENTINELS:
            return None
        return bushmeg_cue_source_weights._normalize_integer(value, name=name, minimum=1)

    setattr(_normalize_optional_positive_integer, _PATCH_ATTR, True)
    _normalize_optional_positive_integer.__wrapped__ = original
    bushmeg_cue_source_weights._normalize_optional_positive_integer = _normalize_optional_positive_integer


__all__ = ["install"]
