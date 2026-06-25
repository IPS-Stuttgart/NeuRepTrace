"""Runtime guardrail for BUSH-MEG source-LOSO ensemble numeric config parsing.

The ensemble workflow historically coerced several user-facing configuration
values with ``int(...)`` or ``float(...)`` at the point of use.  That made YAML
booleans integer-like and silently truncated fractional integer controls such as
``ensemble_top_k: 2.5`` or ``rerank_top_k: 2.5``.  The resulting run could use a
different ensemble size or reranking configuration than the config author
intended.  This patch rejects ambiguous boolean and fractional integer controls
before the expensive LOSO workflow starts while preserving existing integer and
string disable aliases.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

_TARGET_MODULE = "neureptrace.bushmeg_source_loso_ensemble"
_PATCH_MARKER = "_neureptrace_bushmeg_source_loso_ensemble_numeric_patch_installed"
_FINDER_MARKER = "_neureptrace_bushmeg_source_loso_ensemble_numeric_finder"
_DISABLE_RERANK_TEXT = {"", "none", "off", "false", "no"}


def _is_boolean_scalar(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _coerce_integer(value: Any, *, name: str, minimum: int | None = None, nonnegative_rerank: bool = False) -> int:
    if _is_boolean_scalar(value):
        raise ValueError(f"{name} must be an integer.")
    if isinstance(value, Integral):
        normalized = int(value)
    else:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer.") from exc
        if not np.isfinite(number) or number % 1.0 != 0.0:
            raise ValueError(f"{name} must be an integer.")
        normalized = int(number)
    if minimum is not None and normalized < minimum:
        if nonnegative_rerank:
            raise ValueError(f"{name} must be non-negative; use 0 to disable reranking.")
        raise ValueError(f"{name} must be at least {minimum}.")
    return normalized


def _ensure_no_boolean_float(value: Any, *, name: str) -> None:
    if _is_boolean_scalar(value):
        raise ValueError(f"{name} must be a finite floating-point value.")


def _iter_grid_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [token.strip() for token in value.split(",") if token.strip()]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _validate_float_grid_values(value: Any, *, name: str) -> None:
    for item in _iter_grid_values(value):
        if _is_boolean_scalar(item):
            raise ValueError(f"{name} must contain numeric values, not booleans.")


def _patch_module(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, False):
        return

    original_normalize_rerank_top_k = module._normalize_rerank_top_k
    original_normalize_temperature = module._normalize_temperature
    original_parse_float_grid = module._parse_float_grid
    original_fit_stacking_weights = module._fit_stacking_weights
    original_run = module.run_bushmeg_source_loso_ensemble

    def _normalize_rerank_top_k(value: Any) -> int:
        if value is None:
            return module.DEFAULT_RERANK_TOP_K
        if isinstance(value, str) and value.strip().lower().replace("-", "_") in _DISABLE_RERANK_TEXT:
            return 0
        return _coerce_integer(value, name="source_loso.rerank_top_k", minimum=0, nonnegative_rerank=True)

    def _normalize_temperature(value: Any, weighting: str) -> float | None:
        if weighting == "softmax" and value is not None:
            _ensure_no_boolean_float(value, name="source_loso.ensemble_temperature")
        return original_normalize_temperature(value, weighting)

    def _parse_float_grid(value: Any, default: Sequence[float]) -> list[float]:
        _validate_float_grid_values(value, name="Reranker alpha grid")
        return original_parse_float_grid(value, default)

    def _fit_stacking_weights(
        probability_cube: np.ndarray,
        labels: np.ndarray,
        *,
        n_classes: int,
        max_iter: int = module.DEFAULT_STACKING_MAX_ITER,
        learning_rate: float = module.DEFAULT_STACKING_LEARNING_RATE,
        epsilon: float = module.DEFAULT_STACKING_EPSILON,
    ) -> np.ndarray:
        _coerce_integer(max_iter, name="Stacking max_iter")
        _ensure_no_boolean_float(learning_rate, name="Stacking learning_rate")
        _ensure_no_boolean_float(epsilon, name="Stacking epsilon")
        return original_fit_stacking_weights(
            probability_cube,
            labels,
            n_classes=n_classes,
            max_iter=max_iter,
            learning_rate=learning_rate,
            epsilon=epsilon,
        )

    def _validate_runner_config(config: dict[str, Any]) -> None:
        source_loso = module._section(config, "source_loso")
        if "ensemble_top_k" in source_loso:
            _coerce_integer(source_loso["ensemble_top_k"], name="source_loso.ensemble_top_k", minimum=1)
        for key in ("rerank_top_k", "ensemble_rerank_top_k"):
            if key in source_loso:
                _normalize_rerank_top_k(source_loso[key])
        weighting = module._normalize_weighting(source_loso.get("ensemble_weighting", module.DEFAULT_ENSEMBLE_WEIGHTING))
        if weighting == "softmax" and "ensemble_temperature" in source_loso:
            _ensure_no_boolean_float(source_loso["ensemble_temperature"], name="source_loso.ensemble_temperature")
        for key in ("rerank_alpha_grid", "ensemble_rerank_alpha_grid"):
            if key in source_loso:
                _validate_float_grid_values(source_loso[key], name="Reranker alpha grid")
        decoding = module._section(config, "decoding") or {}
        if "max_iter" in decoding:
            _coerce_integer(decoding["max_iter"], name="decoding.max_iter", minimum=1)

    def run_bushmeg_source_loso_ensemble(
        config_path: str | Path,
        *,
        overrides: Sequence[str] | None = None,
        out_path: str | Path | None = None,
        inner_cv_out_path: str | Path | None = None,
        predictions_out_path: str | Path | None = None,
        candidate_summary_out_path: str | Path | None = None,
    ):
        config_path = Path(config_path)
        config = module.apply_overrides(module.load_config(config_path), overrides)
        _validate_runner_config(config)
        return original_run(
            config_path,
            overrides=overrides,
            out_path=out_path,
            inner_cv_out_path=inner_cv_out_path,
            predictions_out_path=predictions_out_path,
            candidate_summary_out_path=candidate_summary_out_path,
        )

    module._normalize_rerank_top_k = _normalize_rerank_top_k
    module._normalize_temperature = _normalize_temperature
    module._parse_float_grid = _parse_float_grid
    module._fit_stacking_weights = _fit_stacking_weights
    module.run_bushmeg_source_loso_ensemble = run_bushmeg_source_loso_ensemble
    setattr(module, _PATCH_MARKER, True)


class _BushmegSourceLosoEnsembleNumericPatchLoader(importlib.abc.Loader):
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


class _BushmegSourceLosoEnsembleNumericPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname != _TARGET_MODULE:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _BushmegSourceLosoEnsembleNumericPatchLoader):
            return spec
        spec.loader = _BushmegSourceLosoEnsembleNumericPatchLoader(spec.loader)
        return spec


def install() -> None:
    """Install stricter source-LOSO ensemble numeric validation."""

    loaded = sys.modules.get(_TARGET_MODULE)
    if loaded is not None:
        _patch_module(loaded)
        return

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _BushmegSourceLosoEnsembleNumericPatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)
