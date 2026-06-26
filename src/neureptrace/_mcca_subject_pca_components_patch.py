"""Runtime guardrail for M-CCA subject PCA component parsing.

``subject_pca_components`` caps the within-subject whitening rank before the
multiset CCA solve. Python and NumPy booleans are integer-like, and bare
``int(...)`` conversion silently truncates finite fractional values. Both cases
can turn a configuration mistake into a valid but unintended low-rank alignment.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

_TARGET_MODULE = "neureptrace.decoding.mcca"
_PATCH_MARKER = "_neureptrace_mcca_subject_pca_components_patch_installed"
_FINDER_MARKER = "_neureptrace_mcca_subject_pca_components_finder"
_ERROR_MESSAGE = "subject_pca_components must be a positive integer, infinity, or None."


def _normalize_subject_pca_components(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{_ERROR_MESSAGE} Boolean values are not valid component counts.")

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(_ERROR_MESSAGE) from exc

    if numeric == float("inf"):
        return float("inf")
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(_ERROR_MESSAGE)

    components = int(numeric)
    if components < 1:
        raise ValueError(_ERROR_MESSAGE)
    return components


def _patch_module(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, False):
        return

    original_fit_subject_prewhitener = module._fit_subject_prewhitener

    @wraps(original_fit_subject_prewhitener)
    def _fit_subject_prewhitener(*args: Any, **kwargs: Any) -> Any:
        if "subject_pca_components" in kwargs:
            kwargs["subject_pca_components"] = _normalize_subject_pca_components(kwargs["subject_pca_components"])
        return original_fit_subject_prewhitener(*args, **kwargs)

    module._fit_subject_prewhitener = _fit_subject_prewhitener
    setattr(module, _PATCH_MARKER, True)


class _MCCASubjectPCAComponentsPatchLoader(importlib.abc.Loader):
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


class _MCCASubjectPCAComponentsPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname != _TARGET_MODULE:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _MCCASubjectPCAComponentsPatchLoader):
            return spec
        spec.loader = _MCCASubjectPCAComponentsPatchLoader(spec.loader)
        return spec


def install() -> None:
    """Install validation for M-CCA subject PCA rank caps."""

    loaded = sys.modules.get(_TARGET_MODULE)
    if loaded is not None:
        _patch_module(loaded)

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _MCCASubjectPCAComponentsPatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)
