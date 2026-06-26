"""Register the Category-2 autoencoder LOSO workflow in all-protocol runs."""

from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping
from functools import wraps
from pathlib import Path
from typing import Any

import pandas as pd

_RUNNER = "category2_autoencoder_loso"
_CATEGORY2_MODULE = "neureptrace.bushmeg_category2_autoencoder_loso"
_MARKER = "_neureptrace_category2_autoencoder_all_protocols_patch"


def _category2_spec(allp: Any, previous: Any | None = None) -> Any:
    return allp.MethodSpec(
        _RUNNER,
        "category2_autoencoder",
        2,
        _RUNNER,
        getattr(previous, "config_updates", {}) if previous is not None else {},
        runnable=True,
        blocked_reason="",
        required_modules=(_CATEGORY2_MODULE,),
    )


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _metadata(
    allp: Any,
    spec: Any,
    method_dir: Path,
    config_path: Path,
    summary_path: Path,
    predictions_path: Path,
    inner_path: Path,
    progress: Any,
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
    *,
    resumed: bool,
    method_timeout_seconds: float | None,
    fold_timeout_seconds: float | None,
) -> dict[str, Any]:
    return {
        **spec.metadata(),
        "method_dir": str(method_dir),
        "method_config": str(config_path),
        "raw_summary_csv": str(summary_path),
        "raw_predictions_csv": str(predictions_path),
        "raw_inner_cv_csv": str(inner_path),
        "status_json": str(progress.status_path),
        "run_log": str(progress.log_path),
        "summary_partial_csv": str(progress.summary_partial_path),
        "predictions_partial_csv": str(progress.predictions_partial_path),
        "inner_partial_csv": str(progress.inner_partial_path),
        "n_configured_participants": allp._participant_count_from_config(config),
        "heavy": bool(settings.get("heavy", False)),
        "enabled": bool(settings.get("enabled", True)),
        "smoke_enabled": bool(settings.get("smoke_enabled", False)),
        "method_timeout_seconds": method_timeout_seconds,
        "fold_timeout_seconds": fold_timeout_seconds,
        "resumed": bool(resumed),
    }


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
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run Category-2 autoencoder LOSO with normal all-protocol bookkeeping."""

    del data_dir, participants  # The dedicated Category-2 runner reads paths/participants from its config file.

    config = copy.deepcopy(config)
    method_dir = Path(method_dir)
    method_dir.mkdir(parents=True, exist_ok=True)
    summary_path = method_dir / "summary.csv"
    predictions_path = method_dir / "predictions.csv"
    inner_path = method_dir / "inner_cv.csv"
    config_path = method_dir / "config.yml"

    previous_status = allp._read_json_if_exists(method_dir / "status.json") if resume else {}
    previous_stage = str(previous_status.get("stage", ""))
    if not resume:
        for path in (
            summary_path,
            predictions_path,
            inner_path,
            config_path,
            method_dir / "status.json",
            method_dir / "run.log",
            method_dir / "summary.partial.csv",
            method_dir / "predictions.partial.csv",
            method_dir / "inner_cv.partial.csv",
        ):
            _remove_if_exists(path)
    elif previous_stage and previous_stage not in {"method_done", "method_failed", "method_skipped"} and not summary_path.exists():
        for path in (
            method_dir / "status.json",
            method_dir / "run.log",
            method_dir / "summary.partial.csv",
            method_dir / "predictions.partial.csv",
            method_dir / "inner_cv.partial.csv",
        ):
            _remove_if_exists(path)
        previous_status = {}
        previous_stage = ""

    progress = allp.MethodProgress(
        method_dir,
        method=spec.method,
        aggregate_callback=aggregate_callback,
        method_timeout_seconds=method_timeout_seconds,
        fold_timeout_seconds=fold_timeout_seconds,
    )
    settings = allp._method_settings(all_protocols_config, spec.method)
    metadata = _metadata(
        allp,
        spec,
        method_dir,
        config_path,
        summary_path,
        predictions_path,
        inner_path,
        progress,
        config,
        settings,
        resumed=False,
        method_timeout_seconds=method_timeout_seconds,
        fold_timeout_seconds=fold_timeout_seconds,
    )

    preserve_terminal_status = bool(resume and previous_stage in {"method_failed", "method_skipped"})
    if preserve_terminal_status:
        raw_summary = allp._read_csv_if_nonempty(progress.summary_partial_path)
        raw_predictions = allp._read_csv_if_nonempty(progress.predictions_partial_path)
        metadata["runnable"] = False
        metadata["status"] = "failed" if previous_stage == "method_failed" else "skipped"
        metadata["resumed"] = True
        metadata["skip_reason"] = previous_status.get("skip_reason", "")
        metadata["blocked_reason"] = previous_status.get("error", previous_status.get("skip_reason", ""))
        metadata["error_type"] = previous_status.get("error_type", "")
        metadata["error"] = previous_status.get("error", "")
        return raw_summary, raw_predictions, metadata

    progress.initialize_artifacts()
    allp._yaml_safe_dump(config_path, config)
    progress.update(
        "configured",
        method_family=spec.method_family,
        protocol_category=int(spec.protocol_category),
        runner=spec.runner,
        method_config=str(config_path),
        n_configured_participants=allp._participant_count_from_config(config),
        max_folds=max_folds,
        method_timeout_seconds=method_timeout_seconds,
        fold_timeout_seconds=fold_timeout_seconds,
    )

    progress.update("checking_requirements", include_heavy=bool(include_heavy))
    available, skip_reason = allp._method_availability(
        spec,
        config,
        settings=settings,
        include_heavy=include_heavy,
        max_folds=max_folds,
    )
    metadata["runnable"] = bool(available)
    metadata["status"] = "runnable" if available else "skipped"
    metadata["skip_reason"] = skip_reason
    if not available:
        metadata["blocked_reason"] = skip_reason
        progress.update("method_skipped", skip_reason=skip_reason)
        return pd.DataFrame(), pd.DataFrame(), metadata

    if resume and summary_path.exists():
        raw_summary = allp._read_csv_if_nonempty(summary_path)
        raw_predictions = allp._read_csv_if_nonempty(predictions_path)
        allp._copy_if_exists(summary_path, progress.summary_partial_path)
        allp._copy_if_exists(predictions_path, progress.predictions_partial_path)
        if not progress.inner_partial_path.exists():
            progress.inner_partial_path.write_text("", encoding="utf-8")
        metadata["resumed"] = True
        progress.update("method_done", resumed=True, n_summary_rows=len(raw_summary), n_prediction_rows=len(raw_predictions))
        return raw_summary, raw_predictions, metadata

    progress.start_method_timeout()
    progress.update("loading_subjects")
    try:
        c2 = importlib.import_module(_CATEGORY2_MODULE)
        raw_summary = c2.run_bushmeg_category2_autoencoder_loso(
            config_path,
            out_path=progress.summary_partial_path,
            predictions_out_path=progress.predictions_partial_path,
        )
    except allp.RunTimeoutError as exc:
        metadata.update(
            status="failed",
            runnable=False,
            timeout_kind=exc.kind,
            timeout_seconds=exc.seconds,
            blocked_reason=str(exc),
            skip_reason=str(exc),
        )
        progress.update(
            "method_failed",
            error_type=type(exc).__name__,
            error=str(exc),
            timeout_kind=exc.kind,
            timeout_seconds=exc.seconds,
        )
        return pd.DataFrame(), pd.DataFrame(), metadata
    except Exception as exc:
        metadata.update(status="failed", runnable=False, blocked_reason=str(exc), skip_reason=str(exc))
        progress.update("method_failed", error_type=type(exc).__name__, error=str(exc))
        return pd.DataFrame(), pd.DataFrame(), metadata

    if raw_summary is None:
        raw_summary = allp._read_csv_if_nonempty(progress.summary_partial_path)
    else:
        raw_summary = pd.DataFrame(raw_summary)
    raw_predictions = allp._read_csv_if_nonempty(progress.predictions_partial_path)
    progress.inner_partial_path.write_text("", encoding="utf-8")
    allp._copy_if_exists(progress.summary_partial_path, summary_path)
    allp._copy_if_exists(progress.predictions_partial_path, predictions_path)
    allp._copy_if_exists(progress.inner_partial_path, inner_path)
    metadata["resumed"] = False
    progress.update("method_done", resumed=False, n_summary_rows=len(raw_summary), n_prediction_rows=len(raw_predictions))
    return raw_summary, raw_predictions, metadata


def install() -> None:
    """Install all-protocol registry and runner support for Category-2 autoencoder LOSO."""

    allp = importlib.import_module("neureptrace.bushmeg_all_protocols")
    if getattr(allp, _MARKER, False):
        return

    original_registry = allp.method_registry
    original_run_method = allp._run_method

    @wraps(original_registry)
    def method_registry(*args: Any, **kwargs: Any) -> dict[str, Any]:
        registry = dict(original_registry(*args, **kwargs))
        registry[_RUNNER] = _category2_spec(allp, registry.get(_RUNNER))
        return registry

    @wraps(original_run_method)
    def run_method(spec: Any, **kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        if getattr(spec, "runner", "") != _RUNNER:
            return original_run_method(spec, **kwargs)
        all_protocols_config = kwargs.pop("all_protocols")
        return _run_category2_loso(allp, spec, all_protocols_config=all_protocols_config, **kwargs)

    allp.method_registry = method_registry
    allp._run_method = run_method
    setattr(allp, _MARKER, True)


__all__ = ["install"]
