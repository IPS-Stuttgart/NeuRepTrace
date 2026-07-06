"""Runtime patch for source feature-roll class-label matching."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_source_roll_label_matching_patched"


class _NanLabel:
    def __hash__(self) -> int:
        return 87178291199

    def __eq__(self, other: Any) -> bool:
        return self is other or _is_nan(other)

    def __repr__(self) -> str:
        return "nan"


_NAN = _NanLabel()


class _CompositeLabel:
    def __init__(self, values: tuple[Any, ...]) -> None:
        self.values = values

    def __hash__(self) -> int:
        return hash(self.values)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _CompositeLabel) and self.values == other.values

    def __repr__(self) -> str:
        return repr(self.values)


def _is_nan(value: Any) -> bool:
    if isinstance(value, np.generic):
        value = value.item()
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def _object_vector(values: Iterable[Any]) -> np.ndarray:
    items = list(values)
    out = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        out[index] = value
    return out


def _dict_key(item: tuple[Any, Any]) -> tuple[str, str, str]:
    key, _value = item
    return (type(key).__module__, type(key).__qualname__, repr(key))


def _normalize_label(value: Any) -> Any:
    if _is_nan(value):
        return _NAN
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=object)
        if array.ndim == 0:
            return _normalize_label(array.item())
        return _CompositeLabel(tuple(_normalize_label(item) for item in array.reshape(-1).tolist()))
    if isinstance(value, list):
        return _CompositeLabel(tuple(_normalize_label(item) for item in value))
    if isinstance(value, tuple):
        return _CompositeLabel(tuple(_normalize_label(item) for item in value))
    if isinstance(value, dict):
        return _CompositeLabel(tuple((key, _normalize_label(item)) for key, item in sorted(value.items(), key=_dict_key)))
    return value


def _restore_label(value: Any) -> Any:
    if value is _NAN:
        return np.nan
    if isinstance(value, _CompositeLabel):
        return tuple(_restore_label(item) for item in value.values)
    return value


def _label_vector(values: Any, *, n_rows: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim == 0 or array.shape[0] != n_rows:
        got = 0 if array.ndim == 0 else array.shape[0]
        raise ValueError(f"{name} must contain one value per feature row: {got} != {n_rows}.")
    if array.ndim == 1:
        rows = array.reshape(n_rows).tolist()
    else:
        width = int(np.prod(array.shape[1:], dtype=np.int64))
        if width < 1:
            raise ValueError(f"{name} must contain one value per feature row.")
        flat = array.reshape(n_rows, width).tolist()
        rows = [row[0] if width == 1 else tuple(row) for row in flat]
    return _object_vector(_normalize_label(row) for row in rows)


def install() -> None:
    """Install robust source feature-roll label normalization."""
    source_roll = importlib.import_module("neureptrace.decoding.source_roll")
    original = source_roll.augment_source_with_feature_roll
    if getattr(original, _PATCH_MARKER, False):
        return

    def source_roll_label_vector(values: Any, *, expected_length: int, name: str) -> np.ndarray:
        return _label_vector(values, n_rows=expected_length, name=name)

    @wraps(original)
    def augment_source_with_feature_roll(source_features: Any, source_labels: Any, *, source_domains: Any = None, config: Any = None):
        features = source_roll._feature_matrix(source_features, name="source_features")
        labels = source_roll_label_vector(source_labels, expected_length=features.shape[0], name="source_labels")
        result = original(features, labels, source_domains=source_domains, config=config)
        restored_labels = _object_vector(_restore_label(label) for label in result.labels.tolist())
        return source_roll.SourceFeatureRollResult(
            result.features,
            restored_labels,
            result.synthetic_mask,
            result.content_indices,
            result.shifts,
            result.metadata,
        )

    setattr(augment_source_with_feature_roll, _PATCH_MARKER, True)
    source_roll._label_vector = source_roll_label_vector
    source_roll.augment_source_with_feature_roll = augment_source_with_feature_roll


__all__ = ["install"]
