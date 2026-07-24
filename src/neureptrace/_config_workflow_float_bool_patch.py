"""Accept tolerant legacy workflow scalar and window config values."""

from __future__ import annotations

import importlib
import math
import os
from collections import defaultdict
from functools import wraps
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

_BOOL_PATCH_MARKER = "_neureptrace_config_workflow_float_bool_patch_installed"
_FLOAT_PAIR_PATCH_MARKER = "_neureptrace_config_workflow_string_pair_patch_installed"
_FLOAT_VALUE_PATCH_MARKER = "_neureptrace_config_workflow_float_value_patch_installed"
_OUTPUT_PATH_PATCH_MARKER = "_neureptrace_config_workflow_output_path_patch_installed"
_BENCHMARK_BOOL_PATCH_MARKER = "_neureptrace_benchmark_manifest_bool_patch_installed"
_BENCHMARK_MISSING_PATCH_MARKER = "_neureptrace_benchmark_manifest_missing_scalar_patch_installed"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}


def _install_domain_importance_bool_config_patch() -> None:
    domain_patch = importlib.import_module("neureptrace._domain_importance_bool_config_patch")
    domain_patch.install()


def _install_benchmark_manifest_bool_patch() -> None:
    benchmark = importlib.import_module("neureptrace.benchmark")

    original_missing = benchmark._missing
    if not getattr(original_missing, _BENCHMARK_MISSING_PATCH_MARKER, False):

        @wraps(original_missing)
        def _missing(value: Any) -> bool:
            if value is None:
                return True
            missing = benchmark.pd.isna(value)
            if isinstance(missing, bool):
                return missing or str(value).strip() == ""
            if hasattr(missing, "item"):
                try:
                    return bool(missing.item()) or str(value).strip() == ""
                except ValueError:
                    pass
            raise ValueError(f"Manifest values must be scalar, got {type(value).__name__}.")

        setattr(_missing, _BENCHMARK_MISSING_PATCH_MARKER, True)
        benchmark._missing = _missing

    original_bool_value = benchmark._bool_value
    if getattr(original_bool_value, _BENCHMARK_BOOL_PATCH_MARKER, False):
        return

    @wraps(original_bool_value)
    def _bool_value(row, column: str, default: bool = False) -> bool:
        value = benchmark._string_value(row, column)
        if value is None:
            return default
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
        raise ValueError(f"Manifest column '{column}' must be a boolean value, got {value!r}.")

    setattr(_bool_value, _BENCHMARK_BOOL_PATCH_MARKER, True)
    benchmark._bool_value = _bool_value


def _string_pair_values(value: str) -> list[str] | None:
    text = value.strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [part.strip() for comma_part in text.split(",") for part in comma_part.split() if part.strip()]


def _contains_boolean_scalar(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return True
        if value.dtype == object and value.size == 1:
            return _contains_boolean_scalar(value.reshape(-1)[0])
    return False


def _validate_distinct_output_paths(config_workflow: Any, kwargs: dict[str, Any]) -> None:
    """Reject active workflow outputs that resolve to the same destination."""

    paths = {
        "outputs.metrics_csv": kwargs.get("out_path"),
        "outputs.calibration_csv": kwargs.get("calibration_out_path"),
        "outputs.observations_csv": kwargs.get("observation_out_path"),
    }
    by_destination: dict[str, list[str]] = defaultdict(list)
    for label, path in paths.items():
        if path is None:
            continue
        destination = os.path.normcase(str(Path(path).resolve(strict=False)))
        by_destination[destination].append(label)

    collisions = {
        destination: labels
        for destination, labels in by_destination.items()
        if len(labels) > 1
    }
    if not collisions:
        return

    details = "; ".join(
        f"{', '.join(labels)} -> {destination}"
        for destination, labels in sorted(collisions.items())
    )
    raise config_workflow.DatasetConfigError(
        "Configured output paths must be distinct; "
        f"conflicting outputs: {details}."
    )


def install() -> None:
    """Install tolerant parsing for generated dataset workflow configs."""

    _install_domain_importance_bool_config_patch()
    _install_benchmark_manifest_bool_patch()
    config_workflow = importlib.import_module("neureptrace.config_workflow")

    original_bool = config_workflow._as_bool
    if not getattr(original_bool, _BOOL_PATCH_MARKER, False):

        @wraps(original_bool)
        def _as_bool(value: Any, *, default: bool = False) -> bool:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, Real):
                numeric = float(value)
                if math.isfinite(numeric) and numeric in {0.0, 1.0}:
                    return bool(value)
            return original_bool(value, default=default)

        setattr(_as_bool, _BOOL_PATCH_MARKER, True)
        config_workflow._as_bool = _as_bool

    original_float_value = config_workflow._as_finite_float
    if not getattr(original_float_value, _FLOAT_VALUE_PATCH_MARKER, False):

        @wraps(original_float_value)
        def _as_finite_float(value: Any, *, name: str) -> float:
            if _contains_boolean_scalar(value):
                raise config_workflow.DatasetConfigError(f"'{name}' must contain finite numeric values.")
            return original_float_value(value, name=name)

        setattr(_as_finite_float, _FLOAT_VALUE_PATCH_MARKER, True)
        config_workflow._as_finite_float = _as_finite_float

    original_float_pair = config_workflow._as_float_pair
    if not getattr(original_float_pair, _FLOAT_PAIR_PATCH_MARKER, False):

        @wraps(original_float_pair)
        def _as_float_pair(value: Any, *, name: str) -> tuple[float, float] | None:
            if isinstance(value, str):
                parsed = _string_pair_values(value)
                if parsed is None:
                    return None
                return original_float_pair(parsed, name=name)
            return original_float_pair(value, name=name)

        setattr(_as_float_pair, _FLOAT_PAIR_PATCH_MARKER, True)
        config_workflow._as_float_pair = _as_float_pair

    original_decode_kwargs = config_workflow._decode_kwargs
    if not getattr(original_decode_kwargs, _OUTPUT_PATH_PATCH_MARKER, False):

        @wraps(original_decode_kwargs)
        def _decode_kwargs(config: Any, *, config_path: Path) -> dict[str, Any]:
            kwargs = original_decode_kwargs(config, config_path=config_path)
            _validate_distinct_output_paths(config_workflow, kwargs)
            return kwargs

        setattr(_decode_kwargs, _OUTPUT_PATH_PATCH_MARKER, True)
        config_workflow._decode_kwargs = _decode_kwargs


__all__ = ["install"]
