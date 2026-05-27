"""Runtime compatibility patch for metadata column-index validation.

FieldTrip metadata column specs use zero-based ``trialinfo`` indices. Python
accepts booleans as integers and negative indices as reverse indexing, so
malformed config values such as ``index: true`` or ``index: -1`` can silently
select the wrong trialinfo column. This patch keeps public APIs stable while
rejecting ambiguous or invalid indices before loading data.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from typing import Any, NoReturn

_PATCH_MARKER = "_neureptrace_metadata_column_validation_patch_installed"


def _fail(error_type: type[Exception], message: str) -> NoReturn:
    raise error_type(message)


def _coerce_metadata_column_index(value: Any, *, error_type: type[Exception] = ValueError) -> int:
    if isinstance(value, bool):
        _fail(error_type, "metadata.columns.index must be a non-negative integer, not a boolean.")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            _fail(error_type, "metadata.columns.index must be a non-negative integer.")
        try:
            index = int(text)
        except ValueError as exc:
            raise error_type(f"metadata.columns.index must be a non-negative integer; got {value!r}.") from exc
    elif isinstance(value, numbers.Integral):
        index = int(value)
    elif isinstance(value, numbers.Real):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            _fail(error_type, f"metadata.columns.index must be a non-negative integer; got {value!r}.")
        index = int(numeric)
    else:
        _fail(error_type, f"metadata.columns.index must be a non-negative integer; got {value!r}.")

    if index < 0:
        _fail(error_type, "metadata.columns.index must be non-negative.")
    return index


def _metadata_columns(config: Mapping[str, Any]) -> list[Any]:
    metadata_config = config.get("metadata", {}) or {}
    if not isinstance(metadata_config, Mapping):
        return []
    columns = metadata_config.get("columns", []) or []
    if isinstance(columns, tuple):
        columns = list(columns)
    if not isinstance(columns, list):
        raise ValueError("metadata.columns must be a list of mappings.")
    return columns


def _validate_metadata_columns_config(config: Mapping[str, Any], *, error_type: type[Exception] = ValueError) -> None:
    try:
        columns = _metadata_columns(config)
    except ValueError as exc:
        raise error_type(str(exc)) from exc
    for column in columns:
        if not isinstance(column, Mapping) or "name" not in column or "index" not in column:
            raise error_type("metadata.columns entries must contain name and index.")
        _coerce_metadata_column_index(column["index"], error_type=error_type)


def install() -> None:
    """Install stricter metadata column-index validation."""

    from neureptrace import dataset_config
    from neureptrace.io import fieldtrip_mat

    if getattr(dataset_config, _PATCH_MARKER, False):
        return

    original_validate_dataset_config = dataset_config.validate_dataset_config

    def validate_dataset_config(
        config: Mapping[str, Any],
        *,
        base_dir: Any = ".",
        check_files: bool = False,
    ) -> list[str]:
        _validate_metadata_columns_config(config, error_type=dataset_config.ConfigValidationError)
        return original_validate_dataset_config(config, base_dir=base_dir, check_files=check_files)

    def _metadata_columns_from_config(config: Mapping[str, Any]) -> tuple[Any, ...]:
        specs = []
        for column in _metadata_columns(config):
            if not isinstance(column, Mapping) or "name" not in column or "index" not in column:
                raise ValueError("metadata.columns entries must contain name and index.")
            specs.append(
                fieldtrip_mat.MetadataColumnSpec(
                    name=str(column["name"]),
                    index=_coerce_metadata_column_index(column["index"]),
                    optional=bool(column.get("optional", False)),
                )
            )
        return tuple(specs)

    validate_dataset_config.__doc__ = original_validate_dataset_config.__doc__
    dataset_config.validate_dataset_config = validate_dataset_config
    fieldtrip_mat._metadata_columns_from_config = _metadata_columns_from_config
    setattr(dataset_config, _PATCH_MARKER, True)
    setattr(fieldtrip_mat, _PATCH_MARKER, True)
