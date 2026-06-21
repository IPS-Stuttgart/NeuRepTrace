"""Compatibility wrapper for source alignment helpers.

The implementation lives in :mod:`neureptrace.decoding.source_alignment_core` so
this public module can apply small benchmark-metadata hotfixes without changing
import paths used by existing scripts/tests.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any as _Any

_core = _import_module("neureptrace.decoding.source_alignment_core")

globals().update(
    {
        name: getattr(_core, name)
        for name in dir(_core)
        if not (name.startswith("__") and name != "__all__")
    }
)

__all__ = list(getattr(_core, "__all__", ()))

_ORIGINAL_STATIC_METADATA = _core.SourceAlignmentConfig.static_metadata


def _static_metadata_with_pseudo_label_benchmark_guard(
    self: "_core.SourceAlignmentConfig",
) -> dict[str, _Any]:
    metadata = dict(_ORIGINAL_STATIC_METADATA(self))
    if self.pseudo_label_target_calibrated:
        metadata["alignment_valid_for_benchmark"] = False
        metadata["alignment_valid_for_strict_source_only"] = False
        metadata["alignment_strict_source_only"] = False
    return metadata


_core.SourceAlignmentConfig.static_metadata = _static_metadata_with_pseudo_label_benchmark_guard
