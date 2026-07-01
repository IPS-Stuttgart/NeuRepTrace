"""Compatibility wrapper for source-only feature clipping.

This package wrapper preserves the public ``neureptrace.decoding.source_clipping``
API while revalidating direct dataclass configs through the normal public config
normalizer.  It exists as a conservative patch layer so legacy imports keep
working without changing caller code.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_IMPL_NAME = "neureptrace.decoding._source_clipping_impl"
_IMPL_PATH = Path(__file__).resolve().parent.parent / "source_clipping.py"
_SPEC = importlib.util.spec_from_file_location(_IMPL_NAME, _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - importlib guard
    raise ImportError(f"Could not load source clipping implementation from {_IMPL_PATH}.")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_IMPL_NAME] = _IMPL
_SPEC.loader.exec_module(_IMPL)

for _name in dir(_IMPL):
    if _name.startswith("__") and _name not in {"__doc__"}:
        continue
    globals()[_name] = getattr(_IMPL, _name)


def _coerce_config(config: SourceFeatureClippingConfig | Mapping[str, Any]) -> SourceFeatureClippingConfig:
    if isinstance(config, SourceFeatureClippingConfig):
        return source_feature_clipping_config(
            lower_quantile=config.lower_quantile,
            upper_quantile=config.upper_quantile,
            copy=config.copy,
        )
    return source_feature_clipping_config(**dict(config))


_IMPL._coerce_config = _coerce_config
globals()["_coerce_config"] = _coerce_config

__all__ = [name for name in globals() if not name.startswith("_")]
