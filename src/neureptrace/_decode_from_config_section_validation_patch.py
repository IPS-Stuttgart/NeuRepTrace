"""Reject malformed decode-from-config sections before defaulting them."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any

_PATCH_MARKER = "_neureptrace_decode_from_config_section_validation_patch_installed"


def _section_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    return dict(value)


def install() -> None:
    """Patch decode-from-config section extraction to reject falsey scalars."""

    from neureptrace import decode_from_config

    original_section = decode_from_config._section
    if getattr(original_section, _PATCH_MARKER, False):
        return

    @wraps(original_section)
    def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
        return _section_mapping(config.get(name, {}), name=str(name))

    setattr(_section, _PATCH_MARKER, True)
    decode_from_config._section = _section


__all__ = ["install"]
