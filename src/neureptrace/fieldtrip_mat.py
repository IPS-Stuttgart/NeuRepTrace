from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import mne
import numpy as np
import pandas as pd
import scipy.io as sio

MatPathElement = str | int

DEFAULT_ROOT_PATH: tuple[MatPathElement, ...] = ("data", 0)
DEFAULT_TRIAL_PATH: tuple[MatPathElement, ...] = ("trial", 0, 0)
DEFAULT_TIME_PATH: tuple[MatPathElement, ...] = ("time", 0, 0)
DEFAULT_LABEL_PATH: tuple[MatPathElement, ...] = ("label", 0)
DEFAULT_TRIALINFO_PATH: tuple[MatPathElement, ...] = ("trialinfo", 0)
DEFAULT_SAMPLEINFO_PATH: tuple[MatPathElement, ...] = ("sampleinfo", 0)
DEFAULT_GRAD_PATH: tuple[MatPathElement, ...] = ("grad", 0)


@dataclass(frozen=True)
class FieldTripRawMatConfig:
    """Configuration for reading FieldTrip raw structs from MATLAB ``.mat`` files."""

    root_path: tuple[MatPathElement, ...] = DEFAULT_ROOT_PATH
    trial_path: tuple[MatPathElement, ...] = DEFAULT_TRIAL_PATH
    time_path: tuple[MatPathElement, ...] = DEFAULT_TIME_PATH
    label_path: tuple[MatPathElement, ...] = DEFAULT_LABEL_PATH
    trialinfo_path: tuple[MatPathElement, ...] | None = DEFAULT_TRIALINFO_PATH
    sampleinfo_path: tuple[MatPathElement, ...] | None = DEFAULT_SAMPLEINFO_PATH
    grad_path: tuple[MatPathElement, ...] | None = DEFAULT_GRAD_PATH
    label_base: int | None = 1
    trim_overlong_labels: bool = True
    ch_type: str = "grad"
    metadata_label_column: str = "condition"
    validate_grad_prefix: bool = True


def _coerce_path(path: Sequence[MatPathElement] | str | None, default: tuple[MatPathElement, ...]) -> tuple[MatPathElement, ...]:
    if path is None:
        return default
    if isinstance(path, str):
        parts: list[MatPathElement] = []
        for part in path.replace("/", ".").replace(",", ".").split("."):
            part = part.strip()
            if not part:
                continue
            try:
                parts.append(int(part))
            except ValueError:
                parts.append(part)
        return tuple(parts)
    return tuple(int(part) if isinstance(part, np.integer) else part for part in path)


def _unwrap_scalar_object(value: Any) -> Any:
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    return value


def _is_struct_array(value: Any) -> bool:
    return isinstance(value, np.ndarray) and value.dtype.names is not None


def _field(value: Any, name: str) -> Any:
    value = _unwrap_scalar_object(value)
    if isinstance(value, dict):
        return value[name]
    if _is_struct_array(value):
        if name not in value.dtype.names:
            raise KeyError(f"MATLAB struct does not contain field {name!r}.")
        return value[name]
    if hasattr(value, name):
        return getattr(value, name)
    raise TypeError(f"Cannot access field {name!r} in object of type {type(value).__name__}.")


def _traverse(value: Any, path: Sequence[MatPathElement]) -> Any:
    current = value
    for element in path:
        if isinstance(element, str):
            current = _field(current, element)
        elif isinstance(element, int):
            current = current[element]
        else:
            raise TypeError(f"Unsupported MATLAB path element {element!r}.")
    return current


def _optional_traverse(value: Any, path: Sequence[MatPathElement] | None) -> Any | None:
    if path is None:
        return None
    try:
        return _traverse(value, path)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _matlab_string(value: Any) -> str:
    value = _unwrap_scalar_object(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            if value.ndim == 0:
                return str(value.item())
            return "".join(str(part) for part in value.ravel()).strip()
        if value.dtype == object and value.size == 1:
            return _matlab_string(value.item())
    return str(value)


def _string_vector(value: Any) -> list[str]:
    value = _unwrap_scalar_object(value)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return [_matlab_string(arr.item())]
    return [_matlab_string(item) for item in arr.ravel(order="F")]


def _cell_vector(value: Any) -> list[Any]:
    value = _unwrap_scalar_object(value)
    arr = np.asarray(value, dtype=object)
    if arr.ndim > 1:
        return [arr[index] for index in np.ndindex(arr.shape)]
    return list(arr)


def _trial_array(value: Any, trial_index: int) -> np.ndarray:
    trial = np.asarray(_unwrap_scalar_object(value), dtype=float)
    if trial.ndim != 2:
        raise ValueError(f"FieldTrip trial {trial_index} must be a 2D channels-by-time array; got shape {trial.shape}.")
    return trial


def _time_vector(value: Any, trial_index: int) -> np.ndarray:
    time = np.asarray(_unwrap_scalar_object(value), dtype=float).ravel()
    if time.ndim != 1 or time.size < 2:
        raise ValueError(f"FieldTrip time vector {trial_index} must contain at least two samples.")
    if np.any(np.diff(time) <= 0):
        raise ValueError(f"FieldTrip time vector {trial_index} must be strictly increasing.")
    return time


def _uniform_sfreq(time: np.ndarray) -> float:
    diffs = np.diff(time)
    sample_interval = float(np.median(diffs))
    if not np.allclose(diffs, sample_interval, rtol=1e-6, atol=1e-12):
        raise ValueError("FieldTrip time vectors must be uniformly sampled to build MNE EpochsArray.")
    return float(1.0 / sample_interval)


def _make_unique(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique = []
    for name in names:
        base = name or "CH"
        count = counts.get(base, 0)
        unique.append(base if count == 0 else f"{base}-{count + 1}")
        counts[base] = count + 1
    return unique


def _as_column_values(value: Any, n_rows: int) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(_unwrap_scalar_object(value))
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.ravel(order="F")
    if arr.ndim == 1:
        if arr.size != n_rows:
            raise ValueError(f"trialinfo has {arr.size} rows but there are {n_rows} trials.")
        return arr
    if arr.ndim == 2:
        if arr.shape[0] != n_rows:
            raise ValueError(f"trialinfo has {arr.shape[0]} rows but there are {n_rows} trials.")
        return arr
    raise ValueError(f"Unsupported trialinfo shape: {arr.shape}.")


def _metadata_from_trialinfo(
    trialinfo: Any | None,
    sampleinfo: Any | None,
    *,
    n_trials: int,
    label_base: int | None,
    metadata_label_column: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    metadata = pd.DataFrame({"trial": np.arange(n_trials, dtype=int)})
    event_codes = np.arange(1, n_trials + 1, dtype=int)

    trialinfo_values = _as_column_values(trialinfo, n_trials) if trialinfo is not None else None
    if trialinfo_values is not None:
        if trialinfo_values.ndim == 1:
            metadata["trialinfo"] = trialinfo_values
            condition = trialinfo_values
            if label_base is not None and np.issubdtype(np.asarray(condition).dtype, np.number):
                condition = np.asarray(condition) - label_base
            metadata[metadata_label_column] = condition
            numeric_trialinfo = np.asarray(trialinfo_values)
            if np.issubdtype(numeric_trialinfo.dtype, np.number) and np.all(np.isfinite(numeric_trialinfo)):
                codes = numeric_trialinfo.astype(int)
                if np.all(codes > 0):
                    event_codes = codes
        else:
            for col_idx in range(trialinfo_values.shape[1]):
                metadata[f"trialinfo_{col_idx}"] = trialinfo_values[:, col_idx]
            first = np.asarray(trialinfo_values[:, 0])
            metadata["trialinfo"] = first
            condition = first - label_base if label_base is not None and np.issubdtype(first.dtype, np.number) else first
            metadata[metadata_label_column] = condition
            if np.issubdtype(first.dtype, np.number) and np.all(np.isfinite(first)):
                codes = first.astype(int)
                if np.all(codes > 0):
                    event_codes = codes

    sampleinfo_values = None if sampleinfo is None else np.asarray(_unwrap_scalar_object(sampleinfo))
    if sampleinfo_values is not None:
        if sampleinfo_values.ndim != 2 or sampleinfo_values.shape[0] != n_trials or sampleinfo_values.shape[1] < 2:
            raise ValueError(f"sampleinfo must have shape (n_trials, >=2); got {sampleinfo_values.shape}.")
        metadata["sample_start"] = sampleinfo_values[:, 0].astype(int)
        metadata["sample_stop"] = sampleinfo_values[:, 1].astype(int)

    return metadata, event_codes


def _grad_field(grad: Any | None, name: str) -> Any | None:
    if grad is None:
        return None
    try:
        return _field(grad, name)
    except (KeyError, TypeError):
        return None


def _validate_grad_prefix(grad: Any | None, labels: list[str], n_channels: int) -> None:
    grad_labels_raw = _grad_field(grad, "label")
    if grad_labels_raw is None:
        return
    grad_labels = _string_vector(grad_labels_raw)
    if len(grad_labels) < n_channels:
        warnings.warn(
            f"FieldTrip grad.label has {len(grad_labels)} labels but trials have {n_channels} channels.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if labels[:n_channels] != grad_labels[:n_channels]:
        warnings.warn(
            "FieldTrip data.label prefix does not match grad.label prefix for the trial channel count.",
            RuntimeWarning,
            stacklevel=2,
        )


def load_fieldtrip_raw_mat_epochs(
    mat_path: str | Path,
    *,
    config: FieldTripRawMatConfig | None = None,
    root_path: Sequence[MatPathElement] | str | None = None,
) -> tuple[mne.EpochsArray, pd.DataFrame]:
    """Load a FieldTrip raw MATLAB struct and expose it as MNE epochs plus metadata.

    The default paths match PyMEGDec-style MATLAB v5 files where the top-level
    variable is ``data`` and ``scipy.io.loadmat(path)["data"][0]`` yields a
    FieldTrip-like raw struct with ``trial``, ``time``, ``label``, ``trialinfo``,
    ``sampleinfo``, and optionally ``grad``.
    """

    config = config or FieldTripRawMatConfig()
    if root_path is not None:
        config = FieldTripRawMatConfig(
            root_path=_coerce_path(root_path, config.root_path),
            trial_path=config.trial_path,
            time_path=config.time_path,
            label_path=config.label_path,
            trialinfo_path=config.trialinfo_path,
            sampleinfo_path=config.sampleinfo_path,
            grad_path=config.grad_path,
            label_base=config.label_base,
            trim_overlong_labels=config.trim_overlong_labels,
            ch_type=config.ch_type,
            metadata_label_column=config.metadata_label_column,
            validate_grad_prefix=config.validate_grad_prefix,
        )

    mat = sio.loadmat(mat_path, squeeze_me=False, struct_as_record=True)
    root = _traverse(mat, config.root_path)
    trial_cell = _traverse(root, config.trial_path)
    time_cell = _traverse(root, config.time_path)
    label_raw = _traverse(root, config.label_path)
    trialinfo = _optional_traverse(root, config.trialinfo_path)
    sampleinfo = _optional_traverse(root, config.sampleinfo_path)
    grad = _optional_traverse(root, config.grad_path)

    trial_items = _cell_vector(trial_cell)
    time_items = _cell_vector(time_cell)
    if len(trial_items) != len(time_items):
        raise ValueError(f"FieldTrip trial count ({len(trial_items)}) does not match time count ({len(time_items)}).")
    if not trial_items:
        raise ValueError("FieldTrip data contains no trials.")

    trials = [_trial_array(trial, trial_index) for trial_index, trial in enumerate(trial_items)]
    times = [_time_vector(time, trial_index) for trial_index, time in enumerate(time_items)]
    n_trials = len(trials)
    n_channels, n_times = trials[0].shape
    first_time = times[0]
    sfreq = _uniform_sfreq(first_time)

    for trial_index, (trial, time) in enumerate(zip(trials, times)):
        if trial.shape != (n_channels, n_times):
            raise ValueError(f"FieldTrip trial {trial_index} has shape {trial.shape}; expected {(n_channels, n_times)}.")
        if time.size != n_times:
            raise ValueError(f"FieldTrip trial {trial_index} has {n_times} samples but its time vector has {time.size} entries.")
        if not np.allclose(time, first_time, rtol=1e-6, atol=1e-12):
            raise ValueError("All FieldTrip time vectors must match to build MNE EpochsArray.")

    labels = _string_vector(label_raw)
    if len(labels) < n_channels:
        raise ValueError(f"FieldTrip data.label has {len(labels)} labels but trials have {n_channels} channels.")
    if len(labels) > n_channels:
        if not config.trim_overlong_labels:
            raise ValueError(f"FieldTrip data.label has {len(labels)} labels but trials have {n_channels} channels.")
        warnings.warn(
            f"Trimming FieldTrip data.label from {len(labels)} to {n_channels} entries to match trial matrices.",
            RuntimeWarning,
            stacklevel=2,
        )
        labels = labels[:n_channels]
    labels = _make_unique(labels)

    if config.validate_grad_prefix:
        _validate_grad_prefix(grad, labels, n_channels)

    metadata, event_codes = _metadata_from_trialinfo(
        trialinfo,
        sampleinfo,
        n_trials=n_trials,
        label_base=config.label_base,
        metadata_label_column=config.metadata_label_column,
    )
    events = np.column_stack(
        [
            metadata["sample_start"].to_numpy(dtype=int) if "sample_start" in metadata.columns else np.arange(n_trials, dtype=int),
            np.zeros(n_trials, dtype=int),
            np.asarray(event_codes, dtype=int),
        ]
    )

    data = np.stack(trials, axis=0)
    info = mne.create_info(labels, sfreq=sfreq, ch_types=config.ch_type)
    epochs = mne.EpochsArray(
        data,
        info,
        events=events,
        tmin=float(first_time[0]),
        metadata=metadata,
        baseline=None,
        verbose="error",
    )
    return epochs, metadata
