"""Keep config-derived dataset names scalar when dataset metadata is unnamed.

The config-driven decoders accept a ``dataset`` section as a mapping.  Some
callers omit ``dataset.name`` because the input files already identify the data
source.  The legacy fallback in the config-to-decoder translation used the
entire ``dataset`` mapping as the fallback value, which can leak a dictionary
into ``dataset_name`` result metadata.  This patch preserves the public helper
surface while replacing only mapping-valued fallbacks with a stable empty
string.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import math
import sys
from collections.abc import Mapping
from functools import wraps
from numbers import Real
from types import ModuleType
from typing import Any

_TARGET_MODULES = {
    "neureptrace.config_workflow",
    "neureptrace.decode_from_config",
}
_PATCH_MARKER = "_neureptrace_dataset_name_config_patch_installed"
_BOOL_PATCH_MARKER = "_neureptrace_config_workflow_numeric_bool_patch_installed"
_FINDER_MARKER = "_neureptrace_dataset_name_config_finder"


def _dataset_name_from_config(config: Any, *, default: str = "") -> str:
    if not isinstance(config, Mapping):
        return default
    dataset = config.get("dataset", {})
    if isinstance(dataset, Mapping):
        value = dataset.get("name", default)
    else:
        value = dataset
    if value is None or isinstance(value, Mapping):
        return default
    return str(value)


def _patch_config_workflow_bool_parser(module: ModuleType) -> None:
    if module.__name__ != "neureptrace.config_workflow":
        return
    original_as_bool = getattr(module, "_as_bool", None)
    if original_as_bool is None or getattr(original_as_bool, _BOOL_PATCH_MARKER, False):
        return

    @wraps(original_as_bool)
    def patched_as_bool(value: Any, *, default: bool = False) -> bool:
        if isinstance(value, Real) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric) and numeric in {0.0, 1.0}:
                return bool(value)
        return original_as_bool(value, default=default)

    setattr(patched_as_bool, _BOOL_PATCH_MARKER, True)
    module._as_bool = patched_as_bool


def _patch_module(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, False):
        return
    _patch_config_workflow_bool_parser(module)
    decode_kwargs = getattr(module, "_decode_kwargs", None)
    if decode_kwargs is None:
        setattr(module, _PATCH_MARKER, True)
        return

    @wraps(decode_kwargs)
    def patched_decode_kwargs(config: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        decoded = decode_kwargs(config, *args, **kwargs)
        if isinstance(decoded.get("dataset_name"), Mapping):
            decoded = dict(decoded)
            decoded["dataset_name"] = _dataset_name_from_config(config)
        return decoded

    module._decode_kwargs = patched_decode_kwargs
    setattr(module, _PATCH_MARKER, True)


class _DatasetNameConfigPatchLoader(importlib.abc.Loader):
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


class _DatasetNameConfigPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname not in _TARGET_MODULES:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _DatasetNameConfigPatchLoader):
            return spec
        spec.loader = _DatasetNameConfigPatchLoader(spec.loader)
        return spec


def install() -> None:
    """Install dataset-name normalization for config-driven decode helpers."""

    for module_name in _TARGET_MODULES:
        loaded = sys.modules.get(module_name)
        if loaded is not None:
            _patch_module(loaded)

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _DatasetNameConfigPatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)
