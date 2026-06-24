"""Runtime guardrail for Category-2 autoencoder numeric config parsing.

YAML booleans are instances of integer-like scalar types in Python.  Without an
explicit guard, values such as ``latent_dim: true`` or ``classifier_c: true`` are
silently coerced to ``1``/``1.0``.  Fractional integer controls can also be
silently truncated by ``int(...)``; for example, ``temporal_bins: 1.5`` becomes
``1``.  That turns misspecified configs into valid but unintended BUSH-MEG
Category-2 autoencoder runs.  This patch preserves the existing parser surface
while rejecting boolean values and fractional integer controls before numeric
coercion.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from types import ModuleType
from typing import Any

import numpy as np

_TARGET_MODULE = "neureptrace.bushmeg_category2_autoencoder_loso"
_PATCH_MARKER = "_neureptrace_bushmeg_category2_autoencoder_config_patch_installed"
_FINDER_MARKER = "_neureptrace_bushmeg_category2_autoencoder_config_finder"


def _is_boolean_scalar(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _is_fractional_integer_value(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number % 1.0 != 0.0)


def _patch_module(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, False):
        return

    original_positive_int = module._positive_int
    original_positive_float = module._positive_float

    def _positive_int(value: Any, *, name: str, minimum: int = 1) -> int:
        if _is_boolean_scalar(value) or _is_fractional_integer_value(value):
            raise ValueError(f"{name} must be an integer.")
        return original_positive_int(value, name=name, minimum=minimum)

    def _positive_float(value: Any, *, name: str, minimum: float = 0.0, inclusive: bool = False) -> float:
        if _is_boolean_scalar(value):
            raise ValueError(f"{name} must be a finite floating-point value.")
        return original_positive_float(value, name=name, minimum=minimum, inclusive=inclusive)

    module._positive_int = _positive_int
    module._positive_float = _positive_float
    setattr(module, _PATCH_MARKER, True)


class _Category2AutoencoderConfigPatchLoader(importlib.abc.Loader):
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


class _Category2AutoencoderConfigPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname != _TARGET_MODULE:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _Category2AutoencoderConfigPatchLoader):
            return spec
        spec.loader = _Category2AutoencoderConfigPatchLoader(spec.loader)
        return spec


def install() -> None:
    """Install boolean and fractional-integer validation for the Category-2 autoencoder config."""

    loaded = sys.modules.get(_TARGET_MODULE)
    if loaded is not None:
        _patch_module(loaded)
        return

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _Category2AutoencoderConfigPatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)
