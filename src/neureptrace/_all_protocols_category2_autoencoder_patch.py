"""Wire Category-2 autoencoder LOSO into the BUSH-MEG all-protocol runner."""

from __future__ import annotations

import copy
import importlib
from functools import wraps
from pathlib import Path
from typing import Any

import pandas as pd

_RUNNER = "category2_autoencoder_loso"
_CATEGORY2_MODULE = "neureptrace.bushmeg_category2_autoencoder_loso"
_PATCH_MARKER = "_neureptrace_category2_autoencoder_all_protocols_installed"


def _category2_spec(all_protocols: Any, previous: Any | None = None) -> Any:
    return all_protocols.MethodSpec(
        _RUNNER,
        "category2_autoencoder",
        2,
        _RUNNER,
        getattr(previous, "config_updates", {}) if previous is not None else {},
        runnable=True,
        blocked_reason="",
        required_modules=(_CATEGORY2_MODULE,),
    )


def _run_category2_loso(
    all_protocols: Any,
    spec: Any,
    *,
    config: dict[str, Any],
    all_protocols_config: dict[str, Any],
    method_dir: str | Path,
    max_folds: int | None,
    resume: bool,
    include_heavy: bool,
    aggregate_callback: Any = None,
    method_timeout_seconds: float | None = None,
    fold_timeout_seconds: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    del all_protocols_config

    c2 = importlib.import_module(_CATEGORY2_MODULE)
    method_dir = Path(method_dir)
    method_dir.mkdir(parents=True, exist_ok=True)
    summary_path = method_dir / "summary.csv"
    predictions_path = method_dir / "predictions.csv"
    inner_path = method_dir / "inner_cv.csv"
    config_path = method_dir / "config.yml"
    progress = all_protocols.MethodProgress(
        method_dir,
        method=spec.method,
        aggregate_callback=aggregate_callback,
        method_timeout_seconds=method_timeout_seconds,
        fold_timeout_seconds=fold_timeout_seconds,
    )

    if not resume:
        for path in (
            summary_path,
            predictions_path,
            inner_path,
            config_path,
            progress.status_path,
            progress.log_path,
            progress.summary_partial_path,
            progress.predictions_partial_path,
            progress.inner_partial_path,
        ):
            if path.exists():
                path.unlink()
    elif summary_path.exists():
        raw_summary = all_protocols._read_csv_if_nonempty(summary_path)
        raw_predictions = all_protocols._read_csv_if_nonempty(predictions_path)
        all_protocols._copy_if_exists(summary_path, progress.summary_partial_path)
        all_protocols._copy_if_exists(predictions_path, progress.predictions_partial_path)
        metadata = _metadata(all_protocols, spec, method_dir, config_path, summary_path, predictions_path, inner_path, config, {}, resumed=True)
        metadata["status"] = "runnable"
        metadata["runnable"] = True
        return raw_summary, raw_predictions, metadata

    progress.initialize_artifacts()
    all_protocols._yaml_safe_dump(config_path, copy.deepcopy(config))
    settings: dict[str, Any] = {}
    metadata = _metadata(all_protocols, spec, method_dir, config_path, summary_path, predictions_path, inner_path, config, settings, resumed=False)
    progress.update(
        "configured",
        method_family=spec.method_family,
        protocol_category=int(spec.protocol_category),
        runner=spec.runner,
        method_config=str(config_path),
        n_configured_participants=all_protocols._participant_count_from_config(config),
        max_folds=max_folds,
        method_timeout_seconds=method_timeout_seconds,
        fold_timeout_seconds=fold_timeout_seconds,
    )
    progress.update("checking_requirements", include_heavy=bool(include_heavy))
    available, skip_reason = all_protocols._method_availability(spec, config, settings=settings, include_heavy=include_heavy, max_folds=max_folds)
    metadata["runnable"] = bool(available)
    metadata["status"] = "runnable" if available else "skipped"
    metadata["skip_reason"] = skip_reason
    if not available:
        metadata["blocked_reason"] = skip_reason
        progress.update("method_skipped", skip_reason=skip_reason)
        return pd.DataFrame(), pd.DataFrame(), metadata

    progress.start_method_timeout()
    progress.update("loading_subjects")
    try:
        c2.run_bushmeg_category2_autoencoder_loso(
            config_path,
            out_path=progress.summary_partial_path,
            predictions_out_path=progress.predictions_partial_path,
        )
    except all_protocols.RunTimeoutError as exc:
        metadata.update(status="failed", runnable=False, timeout_kind=exc.kind, timeout_seconds=exc.seconds, blocked_reason=str(exc), skip_reason=str(exc))
        progress.update("method_failed", error_type=type(exc).__name__, error=str(exc), timeout_kind=exc.kind, timeout_seconds=exc.seconds)
        return pd.DataFrame(), pd.DataFrame(), metadata
    except Exception as exc:
        metadata.update(status="failed", runnable=False, blocked_reason=str(exc))
        progress.update("method_failed", error_type=type(exc).__name__, error=str(exc))
        return pd.DataFrame(), pd.DataFrame(), metadata

    raw_summary = all_protocols._read_csv_if_nonempty(progress.summary_partial_path)
    raw_predictions = all_protocols._read_csv_if_nonempty(progress.predictions_partial_path)
    progress.inner_partial_path.write_text("", encoding="utf-8")
    all_protocols._copy_if_exists(progress.summary_partial_path, summary_path)
    all_protocols._copy_if_exists(progress.predictions_partial_path, predictions_path)
    all_protocols._copy_if_exists(progress.inner_partial_path, inner_path)
    progress.update("method_done", resumed=False, n_summary_rows=len(raw_summary), n_prediction_rows=len(raw_predictions))
    return raw_summary, raw_predictions, metadata


def _metadata(
    all_protocols: Any,
    spec: Any,
    method_dir: Path,
    config_path: Path,
    summary_path: Path,
    predictions_path: Path,
    inner_path: Path,
    config: dict[str, Any],
    settings: dict[str, Any],
    *,
    resumed: bool,
) -> dict[str, Any]:
    return {
        **spec.metadata(),
        "method_dir": str(method_dir),
        "method_config": str(config_path),
        "raw_summary_csv": str(summary_path),
        "raw_predictions_csv": str(predictions_path),
        "raw_inner_cv_csv": str(inner_path),
        "status_json": str(method_dir / "status.json"),
        "run_log": str(method_dir / "run.log"),
        "summary_partial_csv": str(method_dir / "summary.partial.csv"),
        "predictions_partial_csv": str(method_dir / "predictions.partial.csv"),
        "n_configured_participants": all_protocols._participant_count_from_config(config),
        "heavy": bool(settings.get("heavy", False)),
        "enabled": bool(settings.get("enabled", True)),
        "smoke_enabled": bool(settings.get("smoke_enabled", False)),
        "resumed": bool(resumed),
    }


def install() -> None:
    """Install the all-protocol Category-2 autoencoder registration patch."""

    all_protocols = importlib.import_module("neureptrace.bushmeg_all_protocols")
    if getattr(all_protocols, _PATCH_MARKER, False):
        return

    original_registry = all_protocols.method_registry

    @wraps(original_registry)
    def method_registry(*args: Any, **kwargs: Any) -> dict[str, Any]:
        registry = dict(original_registry(*args, **kwargs))
        registry[_RUNNER] = _category2_spec(all_protocols, registry.get(_RUNNER))
        return registry

    original_run_method = all_protocols._run_method

    @wraps(original_run_method)
    def run_method(spec: Any, **kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        if getattr(spec, "runner", "") == _RUNNER:
            all_protocols_config = kwargs.pop("all_protocols")
            return _run_category2_loso(all_protocols, spec, all_protocols_config=all_protocols_config, **kwargs)
        return original_run_method(spec, **kwargs)

    all_protocols.method_registry = method_registry
    all_protocols._run_method = run_method
    setattr(all_protocols, _PATCH_MARKER, True)


__all__ = ["install"]
