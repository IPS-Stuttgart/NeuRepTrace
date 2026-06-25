"""Runtime guardrails for M-CCA configuration parsing.

Boolean scalars are integer-like in Python and NumPy.  Without an explicit
check, values such as ``n_repetitions_per_class=True`` are silently treated as
``1`` by the M-CCA class-repetition alignment helpers.  That can turn a YAML or
programmatic type error into a valid but unintended alignment/calibration run.

The source-alignment and category-2 config builders also expose
``mcca_subject_pca_components`` as an optional nested dimensionality-reduction
setting.  Empty, ``none``, and ``null`` string values should behave like
``None`` rather than being parsed as explicit component requests.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

_TARGET_MODULES = {
    "neureptrace.decoding.mcca",
    "neureptrace.decoding.mcca_target",
    "neureptrace.decoding.source_alignment",
    "neureptrace.decoding.unlabeled_calibration_alignment",
}
_PATCH_MARKER = "_neureptrace_mcca_repetition_count_patch_installed"
_FINDER_MARKER = "_neureptrace_mcca_repetition_count_finder"
_DISABLED_COMPONENT_TEXT_VALUES = {"", "none", "null"}


def _is_boolean_scalar(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _reject_boolean_repetition_count(value: Any) -> None:
    if _is_boolean_scalar(value):
        raise ValueError("n_repetitions_per_class must be a positive integer or None, not a boolean value.")


def _normalize_optional_component_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in _DISABLED_COMPONENT_TEXT_VALUES:
        return None
    return value


def _patch_optional_component_config(module: ModuleType, function_name: str) -> None:
    original_config = getattr(module, function_name)

    @wraps(original_config)
    def config_builder(*args: Any, **kwargs: Any) -> Any:
        if "mcca_subject_pca_components" in kwargs:
            kwargs = dict(kwargs)
            kwargs["mcca_subject_pca_components"] = _normalize_optional_component_value(
                kwargs["mcca_subject_pca_components"]
            )
        return original_config(*args, **kwargs)

    setattr(module, function_name, config_builder)


def _patch_mcca(module: ModuleType) -> None:
    original_class_alignment_matrices = module.class_alignment_matrices
    original_fit_class_mcca = module.fit_class_mcca

    @wraps(original_class_alignment_matrices)
    def class_alignment_matrices(*args: Any, **kwargs: Any) -> Any:
        _reject_boolean_repetition_count(kwargs.get("n_repetitions_per_class"))
        return original_class_alignment_matrices(*args, **kwargs)

    @wraps(original_fit_class_mcca)
    def fit_class_mcca(*args: Any, **kwargs: Any) -> Any:
        _reject_boolean_repetition_count(kwargs.get("n_repetitions_per_class"))
        return original_fit_class_mcca(*args, **kwargs)

    module.class_alignment_matrices = class_alignment_matrices
    module.fit_class_mcca = fit_class_mcca


def _patch_mcca_target(module: ModuleType) -> None:
    original_class_alignment_matrix = module.class_alignment_matrix

    @wraps(original_class_alignment_matrix)
    def class_alignment_matrix(*args: Any, **kwargs: Any) -> Any:
        _reject_boolean_repetition_count(kwargs.get("n_repetitions_per_class"))
        return original_class_alignment_matrix(*args, **kwargs)

    module.class_alignment_matrix = class_alignment_matrix


def _patch_module(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, False):
        return
    if module.__name__ == "neureptrace.decoding.mcca":
        _patch_mcca(module)
    elif module.__name__ == "neureptrace.decoding.mcca_target":
        _patch_mcca_target(module)
    elif module.__name__ == "neureptrace.decoding.source_alignment":
        _patch_optional_component_config(module, "source_alignment_config")
    elif module.__name__ == "neureptrace.decoding.unlabeled_calibration_alignment":
        _patch_optional_component_config(module, "unlabeled_calibration_alignment_config")
    else:  # pragma: no cover - guarded by finder/install targets
        return
    setattr(module, _PATCH_MARKER, True)


class _MCCARepetitionCountPatchLoader(importlib.abc.Loader):
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


class _MCCARepetitionCountPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname not in _TARGET_MODULES:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _MCCARepetitionCountPatchLoader):
            return spec
        spec.loader = _MCCARepetitionCountPatchLoader(spec.loader)
        return spec


def install() -> None:
    """Install validation and normalization for public M-CCA config values."""

    for module_name in _TARGET_MODULES:
        loaded = sys.modules.get(module_name)
        if loaded is not None:
            _patch_module(loaded)

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _MCCARepetitionCountPatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)
