"""Register the Category-2 autoencoder LOSO workflow in all-protocol runs."""

from __future__ import annotations

import importlib
from functools import wraps
from pathlib import Path
from typing import Any

import pandas as pd

_RUNNER = "category2_autoencoder_loso"
_CATEGORY2_MODULE = "neureptrace.bushmeg_category2_autoencoder_loso"
_MARKER = "_neureptrace_category2_autoencoder_all_protocols_patch"


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
        previous = registry.get(_RUNNER)
        registry[_RUNNER] = allp.MethodSpec(
            _RUNNER,
            "category2_autoencoder",
            2,
            _RUNNER,
            getattr(previous, "config_updates", {}) if previous is not None else {},
            runnable=True,
            blocked_reason="",
            required_modules=(_CATEGORY2_MODULE,),
        )
        return registry

    @wraps(original_run_method)
    def run_method(spec: Any, **kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        if getattr(spec, "runner", "") != _RUNNER:
            return original_run_method(spec, **kwargs)

        c2 = importlib.import_module(_CATEGORY2_MODULE)
        config = kwargs["config"]
        all_protocols_config = kwargs["all_protocols"]
        method_dir = Path(kwargs["method_dir"])
        method_dir.mkdir(parents=True, exist_ok=True)
        config_path = method_dir / "config.yml"
        summary_path = method_dir / "summary.csv"
        predictions_path = method_dir / "predictions.csv"
        inner_path = method_dir / "inner_cv.csv"
        summary_partial = method_dir / "summary.partial.csv"
        predictions_partial = method_dir / "predictions.partial.csv"
        inner_partial = method_dir / "inner_cv.partial.csv"

        allp._yaml_safe_dump(config_path, config)
        available, skip_reason = allp._method_availability(
            spec,
            config,
            settings=allp._method_settings(all_protocols_config, spec.method),
            include_heavy=bool(kwargs.get("include_heavy", False)),
            max_folds=kwargs.get("max_folds"),
        )
        metadata = {
            **spec.metadata(),
            "method_dir": str(method_dir),
            "method_config": str(config_path),
            "raw_summary_csv": str(summary_path),
            "raw_predictions_csv": str(predictions_path),
            "raw_inner_cv_csv": str(inner_path),
            "summary_partial_csv": str(summary_partial),
            "predictions_partial_csv": str(predictions_partial),
            "runnable": bool(available),
            "status": "runnable" if available else "skipped",
            "skip_reason": skip_reason,
        }
        if not available:
            return pd.DataFrame(), pd.DataFrame(), metadata

        try:
            summary = c2.run_bushmeg_category2_autoencoder_loso(
                config_path,
                out_path=summary_partial,
                predictions_out_path=predictions_partial,
            )
        except Exception as exc:
            metadata.update(status="failed", runnable=False, blocked_reason=str(exc), skip_reason=str(exc))
            return pd.DataFrame(), pd.DataFrame(), metadata

        predictions = allp._read_csv_if_nonempty(predictions_partial)
        inner_partial.write_text("", encoding="utf-8")
        allp._copy_if_exists(summary_partial, summary_path)
        allp._copy_if_exists(predictions_partial, predictions_path)
        allp._copy_if_exists(inner_partial, inner_path)
        return summary, predictions, metadata

    allp.method_registry = method_registry
    allp._run_method = run_method
    setattr(allp, _MARKER, True)


__all__ = ["install"]
