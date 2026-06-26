"""Runtime guardrails for Protocol 3 calibration/evaluation index vectors.

Protocol 3 uses disjoint same-subject calibration and evaluation rows.  The
core validator historically delegated to ``np.asarray(..., dtype=int)`` and
``np.intersect1d``.  That silently flattened malformed matrix-shaped split
indices and coerced booleans or fractional values into row numbers.  This patch
normalizes only genuine one-dimensional integer index vectors before the
disjointness check runs.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from types import ModuleType
from typing import Any

import numpy as np

_TARGET_MODULE = "neureptrace.bushmeg_all_protocols"
_PATCH_MARKER = "_neureptrace_protocol3_index_validation_patch_installed"
_FINDER_MARKER = "_neureptrace_protocol3_index_validation_finder"


def _as_index_vector(values: Any, *, name: str) -> np.ndarray:
    """Return a validated 1D integer index vector."""

    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim == 1:
        array = array.reshape(-1)
    elif array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    else:
        raise ValueError(f"{name} must be a one-dimensional index vector; got shape {array.shape}.")

    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must contain integer row indices, not boolean values.")
    if array.size == 0:
        return array.astype(int, copy=False)

    try:
        numeric = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer row indices.") from exc

    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain finite integer row indices.")
    if not np.all(np.equal(numeric, np.trunc(numeric))):
        raise ValueError(f"{name} must contain integer row indices.")

    return numeric.astype(int, copy=False)


def _patch_module(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, False):
        return

    def validate_disjoint_calibration_evaluation(calibration_indices: Any, evaluation_indices: Any) -> None:
        calibration = _as_index_vector(calibration_indices, name="calibration_indices")
        evaluation = _as_index_vector(evaluation_indices, name="evaluation_indices")
        overlap = np.intersect1d(calibration, evaluation)
        if overlap.size:
            preview = ",".join(map(str, overlap[:10]))
            raise ValueError(f"Protocol 3 calibration/evaluation rows must be disjoint; overlapping row(s): {preview}.")

    module.validate_disjoint_calibration_evaluation = validate_disjoint_calibration_evaluation
    setattr(module, _PATCH_MARKER, True)


class _Protocol3IndexValidationLoader(importlib.abc.Loader):
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


class _Protocol3IndexValidationFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname != _TARGET_MODULE:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _Protocol3IndexValidationLoader):
            return spec
        spec.loader = _Protocol3IndexValidationLoader(spec.loader)
        return spec


def install() -> None:
    """Install Protocol 3 split-index validation."""

    loaded = sys.modules.get(_TARGET_MODULE)
    if loaded is not None:
        _patch_module(loaded)
        return

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _Protocol3IndexValidationFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)
