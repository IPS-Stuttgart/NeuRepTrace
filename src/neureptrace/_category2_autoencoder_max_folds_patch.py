"""Honor Category-2 autoencoder fold limits in all-protocol smoke runs."""

from __future__ import annotations

import copy
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import Any

_RUNNER_MODULE = "neureptrace.bushmeg_category2_autoencoder_loso"
_ALL_PROTOCOLS_PATCH_MODULE = "neureptrace._category2_autoencoder_all_protocols_patch"
_RUNNER_MARKER = "_neureptrace_category2_autoencoder_max_folds_patch_installed"
_ALL_PROTOCOLS_MARKER = "_neureptrace_category2_autoencoder_max_folds_all_protocols_patch_installed"


class _FoldLimitedSubjectMap(dict[str, Any]):
    """Dict-like subject map whose first sorted iteration exposes only test folds."""

    def __init__(self, subjects: Mapping[str, Any], max_folds: int) -> None:
        super().__init__(subjects)
        self._max_folds = max(1, int(max_folds))
        self._limit_next_iteration = True

    def __iter__(self):  # type: ignore[override]
        if self._limit_next_iteration:
            self._limit_next_iteration = False
            return iter(sorted(super().keys())[: self._max_folds])
        return super().__iter__()


def _resolve_max_folds(module: Any, config: Mapping[str, Any], explicit_max_folds: int | None) -> int | None:
    raw_value: Any = explicit_max_folds
    if raw_value is None:
        section = config.get("category2_autoencoder_loso", {}) or {}
        if isinstance(section, Mapping):
            raw_value = section.get("max_folds", section.get("fold_limit"))
    if raw_value is None:
        return None
    return module._positive_int(raw_value, name="max_folds")


def _resolve_summary_path(module: Any, config_path: Path, config: Mapping[str, Any], out_path: str | Path | None) -> Path:
    if out_path is not None:
        return Path(out_path)
    return module._resolve_output(
        config,
        config_dir=config_path.parent,
        key="category2_autoencoder_loso_summary_csv",
        default="category2_autoencoder_loso_summary.csv",
    )


def _annotate_limited_provenance(summary_path: Path, *, max_folds: int, n_outer_folds: int | None) -> None:
    sidecar = Path(str(summary_path) + ".provenance.json")
    if not sidecar.exists():
        return
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    payload["max_folds"] = int(max_folds)
    if n_outer_folds is not None:
        payload["n_outer_folds"] = int(n_outer_folds)
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _patch_runner_module(module: Any) -> None:
    if getattr(module, _RUNNER_MARKER, False):
        return

    original_run = module.run_bushmeg_category2_autoencoder_loso

    @wraps(original_run)
    def run_bushmeg_category2_autoencoder_loso(
        config_path: str | Path,
        *,
        overrides: Sequence[str] | None = None,
        out_path: str | Path | None = None,
        predictions_out_path: str | Path | None = None,
        max_folds: int | None = None,
    ) -> Any:
        config_path = Path(config_path)
        config = module.apply_overrides(module.load_config(config_path), overrides)
        resolved_max_folds = _resolve_max_folds(module, config, max_folds)
        if resolved_max_folds is None:
            return original_run(config_path, overrides=overrides, out_path=out_path, predictions_out_path=predictions_out_path)

        summary_path = _resolve_summary_path(module, config_path, config, out_path)
        original_loader = module._load_subjects_from_config

        @wraps(original_loader)
        def load_subjects_from_config(*args: Any, **kwargs: Any) -> tuple[Mapping[str, Any], Any]:
            subjects, encoder = original_loader(*args, **kwargs)
            return _FoldLimitedSubjectMap(subjects, resolved_max_folds), encoder

        module._load_subjects_from_config = load_subjects_from_config
        try:
            summary = original_run(config_path, overrides=overrides, out_path=out_path, predictions_out_path=predictions_out_path)
        finally:
            module._load_subjects_from_config = original_loader

        try:
            n_outer_folds = len(summary)
        except TypeError:
            n_outer_folds = None
        _annotate_limited_provenance(summary_path, max_folds=resolved_max_folds, n_outer_folds=n_outer_folds)
        return summary

    module.run_bushmeg_category2_autoencoder_loso = run_bushmeg_category2_autoencoder_loso
    setattr(module, _RUNNER_MARKER, True)


def _with_category2_max_folds(config: Mapping[str, Any], max_folds: int | None) -> Mapping[str, Any]:
    if max_folds is None:
        return config
    updated = copy.deepcopy(dict(config))
    section = updated.setdefault("category2_autoencoder_loso", {})
    if not isinstance(section, dict):
        raise ValueError("Config section 'category2_autoencoder_loso' must be a mapping.")
    section["max_folds"] = int(max_folds)
    return updated


def _patch_all_protocols_patch_module(module: Any) -> None:
    if getattr(module, _ALL_PROTOCOLS_MARKER, False):
        return

    original_run_category2_loso = module._run_category2_loso

    @wraps(original_run_category2_loso)
    def _run_category2_loso(
        allp: Any,
        spec: Any,
        *,
        config: Mapping[str, Any],
        all_protocols_config: Mapping[str, Any],
        method_dir: str | Path,
        data_dir: Any,
        participants: Any,
        max_folds: int | None,
        resume: bool,
        include_heavy: bool,
        aggregate_callback: Any = None,
        method_timeout_seconds: float | None = None,
        fold_timeout_seconds: float | None = None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        _patch_runner_module(importlib.import_module(_RUNNER_MODULE))
        config = _with_category2_max_folds(config, max_folds)
        return original_run_category2_loso(
            allp,
            spec,
            config=config,
            all_protocols_config=all_protocols_config,
            method_dir=method_dir,
            data_dir=data_dir,
            participants=participants,
            max_folds=max_folds,
            resume=resume,
            include_heavy=include_heavy,
            aggregate_callback=aggregate_callback,
            method_timeout_seconds=method_timeout_seconds,
            fold_timeout_seconds=fold_timeout_seconds,
        )

    module._run_category2_loso = _run_category2_loso
    setattr(module, _ALL_PROTOCOLS_MARKER, True)


def install() -> None:
    """Install Category-2 autoencoder fold-limit support."""

    _patch_all_protocols_patch_module(importlib.import_module(_ALL_PROTOCOLS_PATCH_MODULE))
    runner_module = sys.modules.get(_RUNNER_MODULE)
    if runner_module is not None:
        _patch_runner_module(runner_module)


__all__ = ["install"]
