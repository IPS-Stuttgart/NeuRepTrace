"""Runtime patch for stricter few-shot split-index validation."""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from collections.abc import Sequence
from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np


_TARGET_MODULE = "neureptrace.decoding.few_shot"
_PATCH_MARKER = "_neureptrace_few_shot_split_validation_patch_installed"
_FINDER_MARKER = "_neureptrace_few_shot_split_validation_finder"
_INDEX_ERROR = "{name} must contain integer row indices."
_BOOLEAN_INDEX_ERROR = "{name} must contain integer row indices, not booleans or a boolean mask."
_DUPLICATE_INDEX_ERROR = "{name} must not contain duplicate target row indices."


def _normalize_manual_split_indices(values: Sequence[int] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    flat = array.reshape(-1)
    if flat.dtype == np.bool_ or any(isinstance(value, (bool, np.bool_)) for value in flat.tolist()):
        raise ValueError(_BOOLEAN_INDEX_ERROR.format(name=name))
    try:
        numeric = np.asarray(flat, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(_INDEX_ERROR.format(name=name)) from exc
    if not np.all(np.isfinite(numeric)) or not np.all(numeric % 1.0 == 0.0):
        raise ValueError(_INDEX_ERROR.format(name=name))
    indices = numeric.astype(int, copy=False)
    if np.unique(indices).size != indices.size:
        raise ValueError(_DUPLICATE_INDEX_ERROR.format(name=name))
    return indices


def _patch_module(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, False):
        return

    original_select = module.select_few_shot_target_calibration_split
    original_fit = module.fit_few_shot_target_calibrated_decoder

    @wraps(original_select)
    def select_few_shot_target_calibration_split(labels: Any, target_indices: Any = None, *args: Any, **kwargs: Any) -> Any:
        if target_indices is not None:
            target_indices = _normalize_manual_split_indices(target_indices, name="target_indices")
        return original_select(labels, target_indices, *args, **kwargs)

    @wraps(original_fit)
    def fit_few_shot_target_calibrated_decoder(*args: Any, **kwargs: Any) -> Any:
        split = kwargs.get("split")
        if split is not None:
            kwargs = dict(kwargs)
            kwargs["split"] = module.FewShotTargetCalibrationSplit(
                evaluation_indices=_normalize_manual_split_indices(split.evaluation_indices, name="evaluation_indices"),
                calibration_indices=_normalize_manual_split_indices(split.calibration_indices, name="calibration_indices"),
            )
        return original_fit(*args, **kwargs)

    module.select_few_shot_target_calibration_split = select_few_shot_target_calibration_split
    module.fit_few_shot_target_calibrated_decoder = fit_few_shot_target_calibrated_decoder
    setattr(module, _PATCH_MARKER, True)


class _FewShotSplitValidationPatchLoader(importlib.abc.Loader):
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


class _FewShotSplitValidationPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname != _TARGET_MODULE:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _FewShotSplitValidationPatchLoader):
            return spec
        spec.loader = _FewShotSplitValidationPatchLoader(spec.loader)
        return spec


def install() -> None:
    """Install stricter validation for caller-provided few-shot split indices."""

    loaded = sys.modules.get(_TARGET_MODULE)
    if loaded is not None:
        _patch_module(loaded)

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _FewShotSplitValidationPatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)


__all__ = ["install"]