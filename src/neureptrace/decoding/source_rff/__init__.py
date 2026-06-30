"""Compatibility shim for source-RFF helpers.

This package wrapper preserves the historical ``neureptrace.decoding.source_rff``
API while normalizing ``random_state=\"null\"`` like the other optional seed
configuration helpers.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_MODULE_NAME = "_neureptrace_decoding_source_rff_impl"
_MODULE_PATH = Path(__file__).resolve().parent.parent / "source_rff.py"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - importlib guard
    raise ImportError(f"Cannot load source-RFF implementation from {_MODULE_PATH}")

_impl = importlib.util.module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = _impl
_SPEC.loader.exec_module(_impl)

globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})
_original_source_rff_config = _impl.source_rff_config


def _normalize_optional_seed(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def source_rff_config(
    *,
    n_components: Any = _impl.DEFAULT_COMPONENTS,
    gamma: Any = "scale",
    random_state: Any = _impl.DEFAULT_RANDOM_STATE,
    standardize: Any = False,
    epsilon: Any = _impl.DEFAULT_EPSILON,
):
    """Normalize source-RFF options, including ``\"null\"`` optional seeds."""

    return _original_source_rff_config(
        n_components=n_components,
        gamma=gamma,
        random_state=_normalize_optional_seed(random_state),
        standardize=standardize,
        epsilon=epsilon,
    )


_impl.source_rff_config = source_rff_config
globals()["source_rff_config"] = source_rff_config
__all__ = tuple(name for name in globals() if not name.startswith("_"))
