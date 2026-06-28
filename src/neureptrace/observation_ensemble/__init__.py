"""Compatibility wrapper for observation probability ensembling."""

from __future__ import annotations

import sys
from functools import wraps
from importlib import util
from pathlib import Path
from typing import Any

from neureptrace._observation_ensemble_source_debias_bool_patch import normalize_source_baseline_debiasing

_IMPL_NAME = "neureptrace._observation_ensemble_impl"
_IMPL_PATH = Path(__file__).resolve().parents[1] / "observation_ensemble.py"


def _load_impl():
    existing = sys.modules.get(_IMPL_NAME)
    if existing is not None:
        return existing
    spec = util.spec_from_file_location(_IMPL_NAME, _IMPL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load observation ensemble implementation from {_IMPL_PATH}")
    module = util.module_from_spec(spec)
    sys.modules[_IMPL_NAME] = module
    spec.loader.exec_module(module)
    return module


_impl = _load_impl()

for _name in dir(_impl):
    if _name.startswith("__") and _name not in {"__doc__", "__all__"}:
        continue
    globals()[_name] = getattr(_impl, _name)

_original_ensemble_probability_observations = _impl.ensemble_probability_observations


@wraps(_original_ensemble_probability_observations)
def ensemble_probability_observations(*args: Any, **kwargs: Any):
    if "source_baseline_debiasing" in kwargs:
        kwargs = dict(kwargs)
        kwargs["source_baseline_debiasing"] = normalize_source_baseline_debiasing(kwargs["source_baseline_debiasing"])
    return _original_ensemble_probability_observations(*args, **kwargs)


__all__ = tuple(getattr(_impl, "__all__", (name for name in globals() if not name.startswith("_"))))
