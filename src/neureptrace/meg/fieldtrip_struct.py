"""Accessors for MATLAB/FieldTrip-like raw trial structures.

The functions intentionally accept both SciPy ``loadmat`` structured arrays and
plain Python dictionaries.  They are small enough to keep alpha-related code
independent from any particular file-loader or dataset manifest implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

PathToken = str | int
DEFAULT_MAT_ROOT_PATH: tuple[PathToken, ...] = ("data", 0)


def get_data_field(data: Any, field_name: str) -> Any:
    """Return a field from a dict, SciPy MATLAB struct array, or ``np.void`` struct."""

    if isinstance(data, dict):
        return data[field_name]
    if isinstance(data, np.void):
        return data[field_name]

    field = data[field_name]
    if isinstance(field, np.ndarray) and field.size == 1:
        return field.item()
    return field


def has_data_field(data: Any, field_name: str) -> bool:
    """Return whether ``data`` exposes ``field_name``."""

    if isinstance(data, dict):
        return field_name in data
    return bool(getattr(data, "dtype", None) is not None and data.dtype.names and field_name in data.dtype.names)


def unwrap_singleton(value: Any) -> Any:
    """Unwrap nested singleton NumPy arrays created by SciPy's MATLAB reader."""

    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    return value


def value_to_string(value: Any) -> str:
    """Convert MATLAB char/cell string payloads to a Python string."""

    value = unwrap_singleton(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.astype(str).item()
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.kind in {"S", "U"}:
            return "".join(item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in array.ravel())
        if array.dtype == object:
            items = [unwrap_singleton(item) for item in array.ravel()]
            if all(isinstance(item, (bytes, str, np.str_, np.bytes_)) for item in items):
                return "".join(item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in items)
        if array.size == 1:
            return str(array.item())
    return str(value)


def _unwrap_outer_cell_array(cell_array: Any) -> np.ndarray:
    values = np.asarray(cell_array, dtype=object)
    while values.dtype == object and values.size == 1:
        item = values.item()
        item_array = np.asarray(item)
        if not isinstance(item, np.ndarray) or item_array.dtype != object:
            break
        values = np.asarray(item, dtype=object)
    return values


def cell_item(cell_array: Any, index: int) -> Any:
    """Return item ``index`` from a MATLAB cell vector loaded by SciPy."""

    values = _unwrap_outer_cell_array(cell_array)
    if values.ndim == 0:
        return values.item()
    if values.ndim == 2 and values.shape[0] == 1:
        return values[0, index]
    if values.ndim == 2 and values.shape[1] == 1:
        return values[index, 0]
    return values[index]


def count_trials(data: Any) -> int:
    """Return the number of trials in a FieldTrip-like raw structure."""

    trial_field = _unwrap_outer_cell_array(get_data_field(data, "trial"))
    if trial_field.ndim == 2 and trial_field.shape[0] == 1:
        return int(trial_field.shape[1])
    if trial_field.ndim == 2 and trial_field.shape[1] == 1:
        return int(trial_field.shape[0])
    return int(len(trial_field.ravel()))


def get_time_vector(data: Any, trial_idx: int = 0) -> np.ndarray:
    """Return the time vector for one FieldTrip trial."""

    time_vector = cell_item(get_data_field(data, "time"), trial_idx)
    return np.asarray(time_vector, dtype=float).ravel()


def get_trial_signal(data: Any, trial_idx: int = 0) -> np.ndarray:
    """Return one FieldTrip trial as a channels-by-time floating array."""

    trial_signal = cell_item(get_data_field(data, "trial"), trial_idx)
    return np.asarray(trial_signal, dtype=float)


def trial_label(data: Any, trial_idx: int) -> Any:
    """Return one trialinfo label, or ``np.nan`` when trialinfo is unavailable."""

    if not has_data_field(data, "trialinfo"):
        return np.nan
    trialinfo = np.asarray(get_data_field(data, "trialinfo")).ravel()
    if trial_idx >= trialinfo.size:
        return np.nan
    return trialinfo[trial_idx].item()


def follow_path(value: Any, path: Sequence[PathToken]) -> Any:
    """Follow a field/index path through a MAT dictionary or struct."""

    current = value
    for token in path:
        if isinstance(token, str):
            current = get_data_field(current, token) if not isinstance(current, dict) else current[token]
        else:
            current = current[token]
    return current


def parse_path(path: str | Sequence[PathToken] | None, default: Sequence[PathToken] = DEFAULT_MAT_ROOT_PATH) -> tuple[PathToken, ...]:
    """Parse ``"data,0"`` style paths used by command-line tools."""

    if path is None:
        return tuple(default)
    if isinstance(path, str):
        raw_tokens = [token.strip() for token in path.split(",") if token.strip()]
    else:
        raw_tokens = list(path)

    parsed: list[PathToken] = []
    for token in raw_tokens:
        if isinstance(token, int):
            parsed.append(token)
            continue
        text = str(token).strip()
        parsed.append(int(text) if text.lstrip("+-").isdigit() else text)
    return tuple(parsed)


def load_fieldtrip_mat(mat_path: str | Path, *, root_path: str | Sequence[PathToken] | None = DEFAULT_MAT_ROOT_PATH) -> Any:
    """Load a MATLAB v5 file and return the FieldTrip-like root struct."""

    mat = sio.loadmat(Path(mat_path), squeeze_me=False, struct_as_record=True)
    return follow_path(mat, parse_path(root_path, DEFAULT_MAT_ROOT_PATH))
