"""Validate MNE time-decode floating-point time sequences and boolean controls."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_mne_time_decode_float_sequence_validation_patch_installed"
_DEFAULT_NAME = "decode_candidate_times"
_ENSEMBLE_PARSE_PATCH_MARKER = "_neureptrace_ensemble_unique_source_decoder_validation_patch_installed"
_OBSERVATION_ENSEMBLE_PATCH_MARKER = "_neureptrace_observation_unique_decoder_validation_patch_installed"
_BOOL_KWARG_PATCH_MARKER = "_neureptrace_mne_time_decode_bool_kwarg_validation_patch_installed"
_TEMPORAL_TRAIN_WINDOW_PATCH_MARKER = "_neureptrace_temporal_train_window_validation_patch_installed"
_TEMPORAL_TRAIN_WINDOW_KWARG_PATCH_MARKER = "_neureptrace_temporal_train_window_kwarg_validation_patch_installed"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}


def _is_bool_scalar(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _bool_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a boolean value.")


def _normalize_bool(value: Any, *, name: str) -> bool:
    """Normalize YAML/CLI-style boolean tokens while rejecting ambiguous truthiness."""

    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
        raise _bool_error(name)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _bool_error(name)
        return _normalize_bool(value.item(), name=name)
    if isinstance(value, (int, np.integer)):
        if int(value) in {0, 1}:
            return bool(value)
        raise _bool_error(name)
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value) in {0.0, 1.0}:
            return bool(value)
        raise _bool_error(name)
    raise _bool_error(name)


def _normalize_boolean_kwargs(kwargs: dict[str, Any], names: Sequence[str]) -> dict[str, Any]:
    if not any(name in kwargs for name in names):
        return kwargs
    normalized = dict(kwargs)
    for name in names:
        if name in normalized:
            normalized[name] = _normalize_bool(normalized[name], name=name)
    return normalized


def _normalize_temporal_train_window_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    if "temporal_train_window" not in kwargs:
        return kwargs
    normalized = dict(kwargs)
    normalized["temporal_train_window"] = _normalize_temporal_train_window_value(normalized["temporal_train_window"])
    return normalized


def _validation_error(name: str) -> ValueError:
    return ValueError(f"{name} must contain finite numeric time values, not booleans or NaN/inf.")


def _coerce_sequence(value: Any, default: Sequence[float]) -> list[Any]:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return list(default)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return [part.strip() for chunk in text.split(",") for part in chunk.split() if part.strip()]
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return list(value)
    return [value]


def _parse_validated_float_sequence(value: Any, default: Sequence[float], *, name: str) -> tuple[float, ...]:
    values = _coerce_sequence(value, default)
    if any(_is_bool_scalar(item) for item in values):
        raise _validation_error(name)
    try:
        parsed = tuple(float(item) for item in values)
    except (TypeError, ValueError) as exc:
        raise _validation_error(name) from exc
    if not parsed:
        raise ValueError("Expected at least one time value.")
    if not all(np.isfinite(item) for item in parsed):
        raise _validation_error(name)
    return parsed


def _normalize_temporal_train_window_value(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    values = _coerce_sequence(value, ())
    if len(values) != 2:
        raise ValueError("temporal_train_window must contain exactly two finite numeric times: start and stop.")
    if any(_is_bool_scalar(item) for item in values):
        raise _validation_error("temporal_train_window")
    try:
        start, stop = (float(values[0]), float(values[1]))
    except (TypeError, ValueError) as exc:
        raise _validation_error("temporal_train_window") from exc
    if not np.isfinite(start) or not np.isfinite(stop):
        raise _validation_error("temporal_train_window")
    if stop < start:
        raise ValueError("temporal_train_window stop must be greater than or equal to start.")
    return start, stop


def _duplicate_exact_values(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def _decoder_aliases(name: str) -> tuple[str, ...]:
    text = str(name)
    return tuple(dict.fromkeys((text, text.replace("-", "_"), text.replace("_", "-"))))


def _duplicate_alias_pairs(decoders: Sequence[Any]) -> tuple[str, ...]:
    alias_owner: dict[str, str] = {}
    duplicate_pairs: list[str] = []
    for decoder in decoders:
        name = str(decoder)
        aliases = _decoder_aliases(name)
        owner = next((alias_owner[alias] for alias in aliases if alias in alias_owner), None)
        if owner is not None:
            pair = f"{owner!r}/{name!r}"
            if pair not in duplicate_pairs:
                duplicate_pairs.append(pair)
        for alias in aliases:
            alias_owner.setdefault(alias, name)
    return tuple(duplicate_pairs)


def _format_duplicates(duplicates: Sequence[str]) -> str:
    return ", ".join(str(value) for value in duplicates)


def _install_time_sequence_validation() -> None:
    module = importlib.import_module("neureptrace.mne_time_decode")
    original_parse_float_sequence = module._parse_float_sequence
    if getattr(original_parse_float_sequence, _PATCH_MARKER, False):
        return

    @wraps(original_parse_float_sequence)
    def _parse_float_sequence(
        value: object | Sequence[object] | None,
        *,
        default: Sequence[float],
        name: str = _DEFAULT_NAME,
    ) -> tuple[float, ...]:
        normalized_name = str(name).strip() or _DEFAULT_NAME
        return _parse_validated_float_sequence(value, default, name=normalized_name)

    _parse_float_sequence._neureptrace_mne_time_decode_float_sequence_validation_patch_installed = True  # type: ignore[attr-defined]
    module._parse_float_sequence = _parse_float_sequence


def _install_temporal_train_window_validation() -> None:
    module = importlib.import_module("neureptrace.mne_time_decode")
    original_normalize_temporal_train_window = module._normalize_temporal_train_window
    if getattr(original_normalize_temporal_train_window, _TEMPORAL_TRAIN_WINDOW_PATCH_MARKER, False):
        return

    @wraps(original_normalize_temporal_train_window)
    def _normalize_temporal_train_window(temporal_train_window):
        return _normalize_temporal_train_window_value(temporal_train_window)

    setattr(_normalize_temporal_train_window, _TEMPORAL_TRAIN_WINDOW_PATCH_MARKER, True)
    module._normalize_temporal_train_window = _normalize_temporal_train_window


def _wrap_time_decode_boolean_kwargs(module_name: str, parameter_names: Sequence[str]) -> None:
    module = importlib.import_module(module_name)
    original_run_time_resolved_decode = module.run_time_resolved_decode
    if getattr(original_run_time_resolved_decode, _BOOL_KWARG_PATCH_MARKER, False):
        return

    @wraps(original_run_time_resolved_decode)
    def run_time_resolved_decode(*args: Any, **kwargs: Any):
        normalized_kwargs = _normalize_boolean_kwargs(kwargs, parameter_names)
        normalized_kwargs = _normalize_temporal_train_window_kwargs(normalized_kwargs)
        return original_run_time_resolved_decode(*args, **normalized_kwargs)

    setattr(run_time_resolved_decode, _BOOL_KWARG_PATCH_MARKER, True)
    module.run_time_resolved_decode = run_time_resolved_decode


def _install_time_decode_boolean_validation() -> None:
    _wrap_time_decode_boolean_kwargs(
        "neureptrace.mne_time_decode",
        ("label_shuffle_control", "pseudo_label_self_training"),
    )
    _wrap_time_decode_boolean_kwargs(
        "neureptrace.mne_time_decode_foldlocal",
        ("label_shuffle_control",),
    )
    _wrap_time_decode_boolean_kwargs(
        "neureptrace.mne_time_decode_ensemble",
        ("label_shuffle_control", "pseudo_label_self_training", "ensemble_source_baseline_debiasing"),
    )


def _install_time_transfer_temporal_train_window_validation() -> None:
    module = importlib.import_module("neureptrace.time_transfer_decode")
    original_run_time_transfer_decode = module.run_time_transfer_decode
    if getattr(original_run_time_transfer_decode, _TEMPORAL_TRAIN_WINDOW_KWARG_PATCH_MARKER, False):
        return

    @wraps(original_run_time_transfer_decode)
    def run_time_transfer_decode(*args: Any, **kwargs: Any):
        return original_run_time_transfer_decode(*args, **_normalize_temporal_train_window_kwargs(kwargs))

    setattr(run_time_transfer_decode, _TEMPORAL_TRAIN_WINDOW_KWARG_PATCH_MARKER, True)
    module.run_time_transfer_decode = run_time_transfer_decode


def _install_time_decode_ensemble_validation() -> None:
    time_decode_ensemble = importlib.import_module("neureptrace.mne_time_decode_ensemble")
    original_parse_source_decoders = time_decode_ensemble._parse_source_decoders
    if getattr(original_parse_source_decoders, _ENSEMBLE_PARSE_PATCH_MARKER, False):
        return

    @wraps(original_parse_source_decoders)
    def _parse_source_decoders(source_decoders):
        requests, normalized = original_parse_source_decoders(source_decoders)
        duplicates = _duplicate_exact_values(tuple(str(decoder) for decoder in normalized))
        if duplicates:
            raise ValueError(
                "logistic_svm_ensemble source decoders must be unique after alias normalization; "
                f"duplicates: {_format_duplicates(duplicates)}."
            )
        return requests, normalized

    _parse_source_decoders._neureptrace_ensemble_unique_source_decoder_validation_patch_installed = True  # type: ignore[attr-defined]
    time_decode_ensemble._parse_source_decoders = _parse_source_decoders


def _install_observation_ensemble_validation() -> None:
    observation_ensemble = importlib.import_module("neureptrace.observation_ensemble")
    original_ensemble_probability_observations = observation_ensemble.ensemble_probability_observations
    if getattr(original_ensemble_probability_observations, _OBSERVATION_ENSEMBLE_PATCH_MARKER, False):
        return

    @wraps(original_ensemble_probability_observations)
    def ensemble_probability_observations(observations, *args, **kwargs):
        decoders = kwargs.get("decoders", observation_ensemble.DEFAULT_DECODERS)
        if decoders is not None:
            duplicate_pairs = _duplicate_alias_pairs(tuple(decoders))
            if duplicate_pairs:
                raise ValueError(
                    "Ensemble source decoders must be unique after alias normalization; "
                    f"duplicates: {_format_duplicates(duplicate_pairs)}."
                )
        return original_ensemble_probability_observations(observations, *args, **kwargs)

    ensemble_probability_observations._neureptrace_observation_unique_decoder_validation_patch_installed = True  # type: ignore[attr-defined]
    observation_ensemble.ensemble_probability_observations = ensemble_probability_observations


def install() -> None:
    """Patch candidate-time parsing, temporal train-window parsing, boolean controls, and duplicate ensemble-source validation."""

    _install_time_sequence_validation()
    _install_temporal_train_window_validation()
    _install_time_decode_boolean_validation()
    _install_time_transfer_temporal_train_window_validation()
    _install_time_decode_ensemble_validation()
    _install_observation_ensemble_validation()


__all__ = ["install"]
