"""Runtime guardrails for alignment-window numeric config values.

The alignment-window helper historically coerced config attributes with ``float``.
That made YAML booleans valid ``0.0``/``1.0`` values, allowed non-finite window
centers/sizes, and allowed zero or negative sizes. Invalid window metadata can
silently poison cross-window alignment by producing nonsensical time bounds or by
making ``uses_separate_alignment_window`` comparisons unreliable. This patch
keeps the public helper surface while validating numeric scalar inputs before the
``AlignmentWindow`` object is constructed.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from types import ModuleType
from typing import Any

import numpy as np

_TARGET_MODULE = "neureptrace.decoding.alignment_window"
_PATCH_MARKER = "_neureptrace_alignment_window_config_patch_installed"
_FINDER_MARKER = "_neureptrace_alignment_window_config_finder"


def _is_boolean_scalar(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    array = np.asarray(value)
    return array.ndim == 0 and np.issubdtype(array.dtype, np.bool_)


def _validated_float(value: Any, *, name: str, positive: bool = False) -> float:
    if _is_boolean_scalar(value):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be a {qualifier} numeric scalar, not a boolean value.")

    array = np.asarray(value)
    if array.ndim != 0:
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be a {qualifier} numeric scalar.")

    try:
        parsed = float(array)
    except (TypeError, ValueError) as exc:
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be a {qualifier} numeric scalar.") from exc

    if not np.isfinite(parsed) or (positive and parsed <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be a {qualifier} numeric scalar.")
    return parsed


def _resolved_config_value(config: Any, *, override_name: str, fallback_name: str) -> tuple[Any, str]:
    override_value = getattr(config, override_name, None)
    if override_value is not None:
        return override_value, override_name
    try:
        return getattr(config, fallback_name), fallback_name
    except AttributeError as exc:
        raise ValueError(f"{fallback_name} must be provided when {override_name} is not set.") from exc


def _patch_module(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, False):
        return

    alignment_window_cls = module.AlignmentWindow

    def resolved_alignment_window(config: Any):
        center_value, center_name = _resolved_config_value(config, override_name="alignment_window_center", fallback_name="window_center")
        size_value, size_name = _resolved_config_value(config, override_name="alignment_window_size", fallback_name="window_size")
        center = _validated_float(center_value, name=center_name)
        size = _validated_float(size_value, name=size_name, positive=True)
        return alignment_window_cls(center=center, size=size)

    def uses_separate_alignment_window(config: Any) -> bool:
        alignment_window = resolved_alignment_window(config)
        window_center = _validated_float(getattr(config, "window_center"), name="window_center")
        window_size = _validated_float(getattr(config, "window_size"), name="window_size", positive=True)
        return not (np.isclose(alignment_window.center, window_center) and np.isclose(alignment_window.size, window_size))

    module.resolved_alignment_window = resolved_alignment_window
    module.uses_separate_alignment_window = uses_separate_alignment_window
    setattr(module, _PATCH_MARKER, True)


class _AlignmentWindowConfigPatchLoader(importlib.abc.Loader):
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


class _AlignmentWindowConfigPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname != _TARGET_MODULE:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _AlignmentWindowConfigPatchLoader):
            return spec
        spec.loader = _AlignmentWindowConfigPatchLoader(spec.loader)
        return spec


def install() -> None:
    """Install validation for alignment-window config scalar values."""

    loaded = sys.modules.get(_TARGET_MODULE)
    if loaded is not None:
        _patch_module(loaded)
        return

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _AlignmentWindowConfigPatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)
