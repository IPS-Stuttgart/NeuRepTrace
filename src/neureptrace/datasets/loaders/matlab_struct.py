"""MATLAB-struct loader adapter for dataset specs."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from neureptrace.datasets.spec import DatasetFile


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
    if raw_index_path is None and bool(matlab.get("squeeze_first_element", False)):
        raw_index_path = [0]
    index_path = tuple(int(value) for value in (raw_index_path or ()))
    return load_matlab_struct(
        dataset_file.path,
        variable=variable,
        index_path=index_path,
        squeeze_me=bool(matlab.get("squeeze_me", False)),
        struct_as_record=bool(matlab.get("struct_as_record", False)),
    )
