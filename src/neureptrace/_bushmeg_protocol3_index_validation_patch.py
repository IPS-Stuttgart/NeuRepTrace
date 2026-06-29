"""Runtime guardrails for Protocol 3 calibration/evaluation index vectors.

Protocol 3 uses disjoint same-subject calibration and evaluation rows. The
core validator historically delegated to ``np.asarray(..., dtype=int)`` and
``np.intersect1d``. That silently flattened malformed matrix-shaped split
indices and coerced booleans or fractional values into row numbers. This patch
normalizes only genuine one-dimensional integer index vectors before the
disjointness check runs.

The same bool-is-int pitfall also applies to the Protocol 3 split-size options:
``per_class=True`` silently meant one calibration row per class. For YAML-backed
experimental configs that should be rejected explicitly rather than interpreted
as a numeric count or seed.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from functools import wraps
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


def _is_boolean_scalar(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray) and value.shape == () and np.issubdtype(value.dtype, np.bool_):
        return True
    return False


def _reject_boolean_integer_option(value: Any, *, name: str) -> None:
    if _is_boolean_scalar(value):
        raise ValueError(f"{name} must be an integer value, not a boolean value.")


def _patch_module(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, False):
        return

    original_select_split = module.select_bushmeg_target_calibration_split
    original_category3_split = module.category3_calibration_evaluation_split

    def validate_disjoint_calibration_evaluation(calibration_indices: Any, evaluation_indices: Any) -> None:
        calibration = _as_index_vector(calibration_indices, name="calibration_indices")
        evaluation = _as_index_vector(evaluation_indices, name="evaluation_indices")
        overlap = np.intersect1d(calibration, evaluation)
        if overlap.size:
            preview = ",".join(map(str, overlap[:10]))
            raise ValueError(f"Protocol 3 calibration/evaluation rows must be disjoint; overlapping row(s): {preview}.")

    @wraps(original_select_split)
    def select_bushmeg_target_calibration_split(
        target_labels: Any,
        *,
        per_class: Any,
        seed: Any,
        min_evaluation_per_class: Any = 1,
        context: Any = (),
    ) -> Any:
        _reject_boolean_integer_option(per_class, name="per_class")
        _reject_boolean_integer_option(seed, name="seed")
        _reject_boolean_integer_option(min_evaluation_per_class, name="min_evaluation_per_class")
        return original_select_split(
            target_labels,
            per_class=per_class,
            seed=seed,
            min_evaluation_per_class=min_evaluation_per_class,
            context=context,
        )

    @wraps(original_category3_split)
    def category3_calibration_evaluation_split(labels: Any, *, calibration_per_class: Any = 1, seed: Any = 13) -> Any:
        _reject_boolean_integer_option(calibration_per_class, name="calibration_per_class")
        _reject_boolean_integer_option(seed, name="seed")
        return original_category3_split(labels, calibration_per_class=calibration_per_class, seed=seed)

    module.validate_disjoint_calibration_evaluation = validate_disjoint_calibration_evaluation
    module.select_bushmeg_target_calibration_split = select_bushmeg_target_calibration_split
    module.category3_calibration_evaluation_split = category3_calibration_evaluation_split
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
    """Install Protocol 3 split-index and split-count validation."""

    loaded = sys.modules.get(_TARGET_MODULE)
    if loaded is not None:
        _patch_module(loaded)
        return

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _Protocol3IndexValidationFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)
