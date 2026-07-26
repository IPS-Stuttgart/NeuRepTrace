"""Reject malformed decode-from-config sections and preserve section precedence."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any

_SECTION_PATCH_MARKER = "_neureptrace_decode_from_config_section_validation_patch_installed"
_DECODE_KWARGS_PATCH_MARKER = "_neureptrace_decode_from_config_section_precedence_patch_installed"


def _section_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    return dict(value)


def _config_with_authoritative_decoding(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy where legacy workflow fallback is disabled when decoding is present."""

    normalized = dict(config)
    if "decoding" in normalized:
        _section_mapping(normalized.get("decoding"), name="decoding")
        normalized["workflow"] = {}
    return normalized


def install() -> None:
    """Patch section extraction and keep explicit decoding settings authoritative."""

    from neureptrace import decode_from_config

    original_section = decode_from_config._section
    if not getattr(original_section, _SECTION_PATCH_MARKER, False):

        @wraps(original_section)
        def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
            return _section_mapping(config.get(name, {}), name=str(name))

        setattr(_section, _SECTION_PATCH_MARKER, True)
        decode_from_config._section = _section

    original_decode_kwargs = decode_from_config._decode_kwargs
    if not getattr(original_decode_kwargs, _DECODE_KWARGS_PATCH_MARKER, False):

        @wraps(original_decode_kwargs)
        def _decode_kwargs(config: Mapping[str, Any], *, config_dir):
            normalized = _config_with_authoritative_decoding(config)
            return original_decode_kwargs(normalized, config_dir=config_dir)

        setattr(_decode_kwargs, _DECODE_KWARGS_PATCH_MARKER, True)
        decode_from_config._decode_kwargs = _decode_kwargs


__all__ = ["install"]
