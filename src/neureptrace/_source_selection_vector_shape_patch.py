"""Runtime patch for stricter source-selection vector shape validation."""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from collections.abc import Hashable, Sequence
from types import ModuleType
from typing import Any

import numpy as np

_TARGET_MODULE = "neureptrace.decoding.source_selection"
_PATCH_MARKER = "_neureptrace_source_selection_vector_shape_patch_installed"
_FINDER_MARKER = "_neureptrace_source_selection_vector_shape_finder"


def _object_vector_from_array(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array.reshape(-1)
    if array.ndim == 2 and 1 in array.shape:
        return array.reshape(-1)
    raise ValueError(f"{name} must be one-dimensional.")


def _object_vector_from_values(values: Sequence[Any] | np.ndarray, *, name: str, expected_length: int) -> np.ndarray:
    if isinstance(values, np.ndarray):
        vector = _object_vector_from_array(values, name=name)
    elif isinstance(values, (str, bytes)):
        vector = np.asarray([values], dtype=object)
    else:
        try:
            items = list(values)
        except TypeError:
            items = [values]
        vector = np.empty(len(items), dtype=object)
        vector[:] = items

    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one value per source row: {vector.shape[0]} != {expected_length}.")
    return vector


def _domain_vector(values: Sequence[Hashable] | np.ndarray, *, expected_length: int) -> np.ndarray:
    vector = _object_vector_from_values(values, name="source_domains", expected_length=expected_length)
    for domain in vector.tolist():
        try:
            hash(domain)
        except TypeError as exc:
            raise ValueError(f"source_domains must be hashable; got {domain!r}.") from exc
    return vector


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int) -> np.ndarray:
    return _object_vector_from_values(values, name="source_labels", expected_length=expected_length)


def _patch_module(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, False):
        return
    module._domain_vector = _domain_vector
    module._label_vector = _label_vector
    setattr(module, _PATCH_MARKER, True)


class _SourceSelectionVectorShapePatchLoader(importlib.abc.Loader):
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


class _SourceSelectionVectorShapePatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname != _TARGET_MODULE:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _SourceSelectionVectorShapePatchLoader):
            return spec
        spec.loader = _SourceSelectionVectorShapePatchLoader(spec.loader)
        return spec


def install() -> None:
    """Install stricter vector shape validation for source-domain selection."""

    loaded = sys.modules.get(_TARGET_MODULE)
    if loaded is not None:
        _patch_module(loaded)

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _SourceSelectionVectorShapePatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)


__all__ = ["install"]
