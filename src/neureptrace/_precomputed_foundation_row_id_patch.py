"""Preserve composite row ids in precomputed foundation-feature tables."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_precomputed_foundation_row_id_patch_installed"


def _hashable_row_id(value: Any) -> Any:
    """Convert array/list row-id fragments into hashable atomic values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _hashable_row_id(value.item())
        return tuple(_hashable_row_id(item) for item in value.tolist())
    if isinstance(value, list):
        return tuple(_hashable_row_id(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_hashable_row_id(item) for item in value)
    return value


def _row_id_tuple(values: Any, *, expected_length: int | None = None, name: str = "row_ids") -> tuple[Any, ...]:
    """Normalize row ids without flattening row-wise composite identifiers.

    NumPy turns homogeneous ``[(subject, trial), ...]`` row-id lists into a
    rectangular matrix.  Row ids must remain one atomic value per feature row, so
    matrix rows are converted to tuple IDs instead of flattened field-wise.
    """

    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            items = [array.item()]
        elif array.ndim == 1:
            if expected_length == 1 and array.shape[0] != 1:
                items = [tuple(array.tolist())]
            else:
                items = array.tolist()
        else:
            rows = array.reshape(array.shape[0], -1)
            if expected_length is None or rows.shape[0] == expected_length:
                items = [row[0] if row.shape[0] == 1 else tuple(row.tolist()) for row in rows]
            elif expected_length == 1:
                items = [tuple(array.reshape(-1).tolist())]
            elif array.size == expected_length and 1 in array.shape:
                items = array.reshape(-1).tolist()
            else:
                items = [row[0] if row.shape[0] == 1 else tuple(row.tolist()) for row in rows]
    elif isinstance(values, (str, bytes)):
        items = [values]
    else:
        try:
            items = list(values)
        except TypeError:
            items = [values]
        if expected_length == 1 and len(items) != 1:
            items = [tuple(items)]

    row_ids = tuple(_hashable_row_id(item) for item in items)
    if expected_length is not None and len(row_ids) != expected_length:
        raise ValueError(f"{name} must contain one value per feature row: {len(row_ids)} != {expected_length}.")
    return row_ids


def install() -> None:
    """Patch precomputed foundation-feature row-id normalization."""

    module = importlib.import_module("neureptrace.decoding.precomputed_foundation")
    if getattr(module, _PATCH_MARKER, False):
        return

    def __post_init__(self) -> None:
        matrix = module._feature_matrix(self.features, name="features")
        row_ids = _row_id_tuple(self.row_ids, expected_length=matrix.shape[0], name="row_ids")
        feature_names = tuple(self.feature_names)
        if matrix.shape[1] != len(feature_names):
            raise ValueError(f"features and feature_names must have the same number of columns: {matrix.shape[1]} != {len(feature_names)}.")
        module._validate_unique_row_ids(row_ids)
        object.__setattr__(self, "features", matrix.astype(np.float32, copy=False))
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "feature_names", feature_names)

    def _load_npz_features(path: Path, *, features_key: str, row_id_key: str, allow_pickle: bool):
        with np.load(path, allow_pickle=allow_pickle) as payload:
            if features_key in payload:
                features = np.asarray(payload[features_key])
            else:
                matrix_keys = [key for key in payload.files if np.asarray(payload[key]).ndim == 2]
                if len(matrix_keys) != 1:
                    raise ValueError(f"NPZ file must contain key {features_key!r} or exactly one two-dimensional array; found {matrix_keys}.")
                features = np.asarray(payload[matrix_keys[0]])
            features = module._feature_matrix(features, name="features")
            if row_id_key in payload:
                row_ids = _row_id_tuple(payload[row_id_key], expected_length=features.shape[0], name="row ids in NPZ")
            elif "row_id" in payload:
                row_ids = _row_id_tuple(payload["row_id"], expected_length=features.shape[0], name="row ids in NPZ")
            else:
                row_ids = tuple(range(features.shape[0]))
            if "feature_names" in payload:
                feature_names = tuple(str(value) for value in np.asarray(payload["feature_names"], dtype=object).reshape(-1).tolist())
            else:
                feature_names = tuple(f"foundation_{index}" for index in range(features.shape[1]))
        return features, row_ids, feature_names

    def make_precomputed_foundation_feature_table(
        features,
        row_ids=None,
        *,
        feature_names=None,
        feature_fit_scope: str | None = "external_frozen",
        source_model: str = "external",
    ):
        matrix = module._feature_matrix(features, name="features")
        ids = tuple(range(matrix.shape[0])) if row_ids is None else _row_id_tuple(row_ids, expected_length=matrix.shape[0], name="row_ids")
        names = tuple(f"foundation_{index}" for index in range(matrix.shape[1])) if feature_names is None else tuple(str(name) for name in feature_names)
        metadata = module._feature_table_metadata(
            path=None,
            source_model=source_model,
            feature_fit_scope=module.normalize_feature_fit_scope(feature_fit_scope),
            n_rows=matrix.shape[0],
            n_features=matrix.shape[1],
        )
        return module.PrecomputedFoundationFeatureTable(features=matrix, row_ids=ids, feature_names=names, metadata=metadata)

    def align_precomputed_foundation_features(table, row_ids):
        requested = _row_id_tuple(row_ids, name="row_ids")
        index = table.row_index()
        missing = [row_id for row_id in requested if row_id not in index]
        if missing:
            preview = ", ".join(repr(row_id) for row_id in missing[:5])
            raise KeyError(f"Precomputed feature table is missing {len(missing)} requested row id(s): {preview}.")
        return table.features[[index[row_id] for row_id in requested]].astype(np.float32, copy=False)

    module._row_id_tuple = _row_id_tuple
    module.PrecomputedFoundationFeatureTable.__post_init__ = __post_init__
    module._load_npz_features = _load_npz_features
    module.make_precomputed_foundation_feature_table = make_precomputed_foundation_feature_table
    module.align_precomputed_foundation_features = align_precomputed_foundation_features
    setattr(module, _PATCH_MARKER, True)


__all__ = ["install"]
