"""Runtime guardrails for LoRA few-shot numeric and probability parsing.

Python/YAML booleans are numeric scalar types, so float validators would otherwise
coerce values such as ``true``/``false`` to ``1.0``/``0.0``.  That can silently
turn misspecified LoRA few-shot configs into valid but unintended runs.

The patch also validates LoRA probability matrices before row-wise normalization.
Without the explicit shape check, a one-dimensional malformed probability vector
raises a NumPy axis error instead of the public ValueError used by the other
few-shot probability helpers.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

_TARGET_MODULES = (
    "neureptrace.decoding.lora_few_shot",
    "neureptrace.decoding.semi_supervised_lora_few_shot",
)
_PATCH_MARKER = "_neureptrace_lora_few_shot_numeric_config_patch_installed"
_FINDER_MARKER = "_neureptrace_lora_few_shot_numeric_config_finder"
_PROBABILITY_NORMALIZE_MARKER = "_neureptrace_lora_probability_rows_wrapped"


def _is_boolean_scalar(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _boolean_float_error_message(validator_name: str, name: str, kwargs: dict[str, Any]) -> str:
    if validator_name == "_positive_float":
        return f"{name} must be a positive finite value."
    if validator_name == "_nonnegative_float":
        return f"{name} must be a non-negative finite value."
    lower = kwargs.get("lower")
    upper = kwargs.get("upper")
    return f"{name} must be in [{lower}, {upper}]."


def _patch_float_validator(module: ModuleType, validator_name: str) -> None:
    original = getattr(module, validator_name)

    def _wrapped(value: Any, *args: Any, **kwargs: Any):
        if _is_boolean_scalar(value):
            name = kwargs.get("name")
            if name is None and args:
                name = args[0]
            if name is None:
                name = "value"
            raise ValueError(_boolean_float_error_message(validator_name, str(name), kwargs))
        return original(value, *args, **kwargs)

    setattr(module, validator_name, _wrapped)


def _patch_probability_normalizer(module: ModuleType) -> None:
    original = getattr(module, "_normalize_probability_rows", None)
    if original is None or getattr(original, _PROBABILITY_NORMALIZE_MARKER, False):
        return

    @wraps(original)
    def _normalize_probability_rows(probabilities: Any):
        matrix = np.asarray(probabilities, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Predicted probabilities must be finite two-dimensional rows with positive mass.")
        return original(matrix)

    setattr(_normalize_probability_rows, _PROBABILITY_NORMALIZE_MARKER, True)
    setattr(module, "_normalize_probability_rows", _normalize_probability_rows)


def _patch_module(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, False):
        return

    for validator_name in ("_positive_float", "_nonnegative_float", "_bounded_float"):
        if hasattr(module, validator_name):
            _patch_float_validator(module, validator_name)
    _patch_probability_normalizer(module)
    setattr(module, _PATCH_MARKER, True)


class _LoRAFewShotNumericConfigPatchLoader(importlib.abc.Loader):
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


class _LoRAFewShotNumericConfigPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname not in _TARGET_MODULES:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _LoRAFewShotNumericConfigPatchLoader):
            return spec
        spec.loader = _LoRAFewShotNumericConfigPatchLoader(spec.loader)
        return spec


def install() -> None:
    """Install boolean-value validation for LoRA few-shot float hyperparameters."""

    for target_module in _TARGET_MODULES:
        loaded = sys.modules.get(target_module)
        if loaded is not None:
            _patch_module(loaded)

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _LoRAFewShotNumericConfigPatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)
