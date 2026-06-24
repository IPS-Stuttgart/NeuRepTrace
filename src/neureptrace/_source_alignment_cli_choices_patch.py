"""Keep source-alignment CLI choices synchronized after runtime patches."""

from __future__ import annotations

import sys
from types import ModuleType

_TARGET_MODULE = "neureptrace.decoding.source_alignment"
_PATCH_MARKER = "_neureptrace_source_alignment_cli_choices_patch_installed"
_DOWNSTREAM_MODULES = (
    "neureptrace.mne_time_decode",
    "neureptrace.mne_time_decode_ensemble",
)


def _source_alignment_run_methods(source_alignment: ModuleType) -> tuple[str, ...]:
    methods = getattr(source_alignment, "SOURCE_ALIGNMENT_METHODS")
    return tuple(dict.fromkeys((*methods, "off", "raw")))


def _patch_module(module: ModuleType, source_alignment: ModuleType) -> None:
    if not hasattr(module, "SOURCE_ALIGNMENT_RUN_METHODS"):
        return
    module.SOURCE_ALIGNMENT_RUN_METHODS = _source_alignment_run_methods(source_alignment)
    setattr(module, _PATCH_MARKER, True)


def install() -> None:
    """Refresh already-imported time-decode modules after late alignment extensions."""

    source_alignment = sys.modules.get(_TARGET_MODULE)
    if not isinstance(source_alignment, ModuleType):
        return
    for module_name in _DOWNSTREAM_MODULES:
        module = sys.modules.get(module_name)
        if isinstance(module, ModuleType):
            _patch_module(module, source_alignment)


__all__ = ["install"]
