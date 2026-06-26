"""Runtime guardrail for M-CCA common component-count parsing.

``n_components`` controls the number of common M-CCA dimensions retained after
the shared-space SVD. Python and NumPy boolean scalars are integer-like, so the
existing numeric normalization can silently interpret ``True`` as one retained
component. That can turn a YAML or programmatic type error into a valid but
unintended one-dimensional alignment.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

_TARGET_MODULE = "neureptrace.decoding.mcca"
_PATCH_MARKER = "_neureptrace_mcca_component_count_patch_installed"
_FINDER_MARKER = "_neureptrace_mcca_component_count_finder"
_ERROR_MESSAGE = "n_components must be a positive integer component count or infinity."


def _reject_boolean_component_count(value: Any) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{_ERROR_MESSAGE} Boolean values are not valid component counts.")


def _patch_module(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, False):
        return

    original_requested_component_count = module._requested_component_count

    @wraps(original_requested_component_count)
    def _requested_component_count(n_components: Any) -> int:
        _reject_boolean_component_count(n_components)
        return original_requested_component_count(n_components)

    module._requested_component_count = _requested_component_count
    setattr(module, _PATCH_MARKER, True)


class _MCCAComponentCountPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader: importlib.abc.Loader) -> None:
        self.wrapped_loader = wrapped_loader

    def create_module(self, spec):  # type: ignore[override]
        create_module = getattr(self.wrapped_loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped_loader.exec_module(module)
        _patch_module(module)

    def get_code(self, fullname: str):
        get_code = getattr(self.wrapped_loader, "get_code", None)
        if get_code is None:
            raise ImportError(f"Loader for {fullname!r} does not provide executable code.")
        return get_code(fullname)

    def get_source(self, fullname: str):
        get_source = getattr(self.wrapped_loader, "get_source", None)
        if get_source is None:
            return None
        return get_source(fullname)

    def is_package(self, fullname: str) -> bool:
        is_package = getattr(self.wrapped_loader, "is_package", None)
        if is_package is None:
            return False
        return bool(is_package(fullname))


class _MCCAComponentCountPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname != _TARGET_MODULE:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _MCCAComponentCountPatchLoader):
            return spec
        spec.loader = _MCCAComponentCountPatchLoader(spec.loader)
        return spec


def install() -> None:
    """Install validation for public M-CCA component-count requests."""

    loaded = sys.modules.get(_TARGET_MODULE)
    if loaded is not None:
        _patch_module(loaded)

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _MCCAComponentCountPatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)
