"""Canonicalize one-pass iterators in observation provenance hashes."""

from __future__ import annotations

from collections.abc import Iterator

_PATCH_MARKER = "_neureptrace_stable_hash_iterator_patch_installed"


def install() -> None:
    """Make equivalent iterator-backed provenance payloads hash identically."""

    from neureptrace import observations

    if getattr(observations, _PATCH_MARKER, False):
        return

    original_default = observations._stable_json_default

    def _stable_json_default(value: object) -> object:
        if isinstance(value, Iterator):
            return list(value)
        return original_default(value)

    observations._stable_json_default = _stable_json_default
    setattr(observations, _PATCH_MARKER, True)


__all__ = ["install"]
