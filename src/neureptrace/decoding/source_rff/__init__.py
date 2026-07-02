"""Compatibility shim for source-RFF helpers.

This package wrapper preserves the historical ``neureptrace.decoding.source_rff``
API while normalizing text-null optional random-state values like the other
source preprocessing helpers.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

_MODULE_NAME = "_neureptrace_decoding_source_rff_impl"
_MODULE_PATH = Path(__file__).resolve().parent.parent / "source_rff.py"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - importlib guard
    raise ImportError(f"Cannot load source-RFF implementation from {_MODULE_PATH}")

_impl = importlib.util.module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = _impl
_SPEC.loader.exec_module(_impl)

_original_optional_random_state = _impl._optional_random_state
_original_source_rff_config = _impl.source_rff_config


def _is_none_like_optional_seed(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"})


def _normalize_optional_seed(value: Any) -> Any:
    if _is_none_like_optional_seed(value):
        return None
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return value
        value = value.item()
        if _is_none_like_optional_seed(value):
            return None
    if isinstance(value, np.generic):
        value = value.item()
        if _is_none_like_optional_seed(value):
            return None
    return value


def _optional_random_state(value: Any) -> int | None:
    """Normalize optional source-RFF random-state values before validation."""

    return _original_optional_random_state(_normalize_optional_seed(value))


_impl._optional_random_state = _optional_random_state

globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})


def source_rff_config(
    *,
    n_components: Any = _impl.DEFAULT_COMPONENTS,
    gamma: Any = "scale",
    random_state: Any = _impl.DEFAULT_RANDOM_STATE,
    standardize: Any = False,
    epsilon: Any = _impl.DEFAULT_EPSILON,
):
    """Normalize source-RFF options, including text-null optional seeds."""

    return _original_source_rff_config(
        n_components=n_components,
        gamma=gamma,
        random_state=_normalize_optional_seed(random_state),
        standardize=standardize,
        epsilon=epsilon,
    )


_impl.source_rff_config = source_rff_config
globals()["_optional_random_state"] = _optional_random_state
globals()["source_rff_config"] = source_rff_config
__all__ = tuple(name for name in globals() if not name.startswith("_"))
