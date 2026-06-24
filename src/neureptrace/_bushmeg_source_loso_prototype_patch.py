"""Runtime patch for BUSH-MEG source-LOSO prototype base features.

The source-LOSO module accepts ``evoked_dct_prototype``, but its base-feature
lookup omitted that entry.  Because the core helper falls back to ``evoked`` for
unmapped prototype kinds, DCT prototype candidates were evaluated with evoked
bin means.  This patch extends the map once the target module is loaded.
"""

from __future__ import annotations

import importlib.abc
import sys
from types import ModuleType
from typing import Any

_TARGET_MODULE = "neureptrace.bushmeg_source_loso"
_PATCH_MARKER = "_neureptrace_bushmeg_source_loso_prototype_patch_installed"
_FINDER_MARKER = "_neureptrace_bushmeg_source_loso_prototype_patch_finder"
_PROTOTYPE_BASE_FIXES = {"evoked_dct_prototype": "evoked_dct"}


def _patch_module(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, False):
        return
    mapping = getattr(module, "PROTOTYPE_BASE_FEATURE_KINDS", None)
    if mapping is None:
        raise AttributeError("bushmeg_source_loso.PROTOTYPE_BASE_FEATURE_KINDS is not available.")
    mapping.update(_PROTOTYPE_BASE_FIXES)
    setattr(module, _PATCH_MARKER, True)


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader):
        self._wrapped = wrapped

    def create_module(self, spec: Any) -> ModuleType | None:
        create_module = getattr(self._wrapped, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        exec_module = getattr(self._wrapped, "exec_module", None)
        if exec_module is None:
            raise ImportError(f"Cannot patch {_TARGET_MODULE}: loader does not implement exec_module.")
        exec_module(module)
        _patch_module(module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: ModuleType | None = None) -> Any:
        if fullname != _TARGET_MODULE:
            return None
        for finder in tuple(sys.meta_path):
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                continue
            spec.loader = _PatchLoader(spec.loader)
            return spec
        return None


def install() -> None:
    module = sys.modules.get(_TARGET_MODULE)
    if isinstance(module, ModuleType):
        _patch_module(module)
        return
    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _PatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)
