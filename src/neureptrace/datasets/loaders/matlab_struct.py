"""MATLAB-struct loader adapter for dataset specs."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

from neureptrace.datasets.spec import DatasetFile

_TRUE_BOOL_TOKENS = {"1", "true", "yes", "on"}
_FALSE_BOOL_TOKENS = {"", "0", "false", "no", "off", "none", "null"}


def _normalize_bool(raw_value: Any, *, name: str) -> bool:
    """Normalize dataset-spec booleans without treating non-empty strings as true."""

    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return False
    if isinstance(raw_value, Integral):
        if int(raw_value) in {0, 1}:
            return bool(raw_value)
        raise ValueError(f"{name} must be a boolean value, not {raw_value!r}.")
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in _TRUE_BOOL_TOKENS:
            return True
        if normalized in _FALSE_BOOL_TOKENS:
            return False
        raise ValueError(f"{name} must be a boolean value, not {raw_value!r}.")
    raise ValueError(f"{name} must be a boolean value, not {type(raw_value).__name__}.")


def _normalize_index_path(raw_index_path: Any) -> tuple[int, ...]:
    """Normalize ``matlab.index_path`` without lossy truthiness or string iteration."""

    if raw_index_path is None:
        return ()
    if isinstance(raw_index_path, (str, bytes)):
        raise ValueError("matlab.index_path must be an integer or a sequence of integer indices, not a string.")
    if isinstance(raw_index_path, bool):
        raise ValueError("matlab.index_path entries must be integer indices, not booleans.")
    if isinstance(raw_index_path, Integral):
        return (int(raw_index_path),)
    if not isinstance(raw_index_path, Sequence):
        raise ValueError("matlab.index_path must be an integer or a sequence of integer indices.")

    normalized: list[int] = []
    for value in raw_index_path:
        if isinstance(value, bool):
            raise ValueError("matlab.index_path entries must be integer indices, not booleans.")
        if not isinstance(value, Integral):
            raise ValueError("matlab.index_path entries must be integer indices.")
        normalized.append(int(value))
    return tuple(normalized)


def load_matlab_struct(
    path: str | os.PathLike[str],
    *,
    variable: str = "data",
    index_path: Sequence[int] = (),
    squeeze_me: bool = False,
    struct_as_record: bool = False,
) -> Any:
    """Load a variable from a MATLAB file and optionally index into nested arrays."""

    try:
        import scipy.io as sio
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency is declared by the package.
        raise RuntimeError("The MATLAB struct loader requires scipy.") from exc

    mat = sio.loadmat(Path(path), squeeze_me=squeeze_me, struct_as_record=struct_as_record)
    if variable not in mat:
        available = ", ".join(sorted(key for key in mat if not key.startswith("__")))
        raise KeyError(f"MATLAB variable '{variable}' is missing from {path}. Available variables: {available}")

    value: Any = mat[variable]
    for index in index_path:
        value = value[index]
    return value


def load_matlab_struct_recording(dataset_file: DatasetFile, spec: Mapping[str, Any]) -> Any:
    """Load a resolved dataset file according to the spec's ``matlab`` section."""

    matlab = spec.get("matlab", {})
    if not isinstance(matlab, Mapping):
        raise ValueError("matlab section must be a mapping when present")

    variable = str(matlab.get("variable", "data"))
    raw_index_path = matlab.get("index_path")
    if raw_index_path is None and _normalize_bool(matlab.get("squeeze_first_element", False), name="matlab.squeeze_first_element"):
        raw_index_path = [0]
    index_path = _normalize_index_path(raw_index_path)
    return load_matlab_struct(
        dataset_file.path,
        variable=variable,
        index_path=index_path,
        squeeze_me=_normalize_bool(matlab.get("squeeze_me", False), name="matlab.squeeze_me"),
        struct_as_record=_normalize_bool(matlab.get("struct_as_record", False), name="matlab.struct_as_record"),
    )
