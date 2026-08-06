"""Canonicalize iterators and NumPy arrays in observation provenance hashes."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import numpy as np

_PATCH_MARKER = "_neureptrace_stable_hash_iterator_patch_installed"


def _numpy_array_payload(value: np.ndarray) -> dict[str, object]:
    """Return a content-complete, print-option-independent array payload."""

    array = np.asarray(value)
    payload: dict[str, object] = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }
    if array.dtype.hasobject:
        payload["values"] = array.tolist()
    else:
        contiguous = np.ascontiguousarray(array)
        payload["content_sha256"] = hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()
    return {"__numpy_array__": payload}


def install() -> None:
    """Make equivalent iterator- and array-backed payloads hash deterministically."""

    from neureptrace import observations

    if getattr(observations, _PATCH_MARKER, False):
        return

    original_default = observations._stable_json_default

    def _stable_json_default(value: object) -> object:
        if isinstance(value, np.ndarray):
            return _numpy_array_payload(value)
        if isinstance(value, Iterator):
            return list(value)
        return original_default(value)

    observations._stable_json_default = _stable_json_default
    setattr(observations, _PATCH_MARKER, True)


__all__ = ["install"]
