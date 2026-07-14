"""Runtime patch for source feature-roll labels and output precision."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_source_roll_label_matching_patched"
_ROLL_PRECISION_MARKER = "_source_roll_feature_precision_patched"


class _NanLabel:
    def __hash__(self) -> int:
        return 87178291199

    def __eq__(self, other: Any) -> bool:
        return self is other or _is_nan(other)

    def __repr__(self) -> str:
        return "nan"


_NAN = _NanLabel()


class _TemporalNaTLabel:
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def __hash__(self) -> int:
        return hash(("neureptrace-temporal-nat", self.kind))

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _TemporalNaTLabel) and self.kind == other.kind

    def __repr__(self) -> str:
        return f"{self.kind}('NaT')"


_DATETIME_NAT = _TemporalNaTLabel("datetime64")
_TIMEDELTA_NAT = _TemporalNaTLabel("timedelta64")


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


def _temporal_nat_label(value: Any) -> _TemporalNaTLabel | None:
    if isinstance(value, np.datetime64) and bool(np.isnat(value)):
        return _DATETIME_NAT
    if isinstance(value, np.timedelta64) and bool(np.isnat(value)):
        return _TIMEDELTA_NAT
    return None


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
    temporal_nat = _temporal_nat_label(value)
    if temporal_nat is not None:
        return temporal_nat
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
    if value is _DATETIME_NAT:
        return np.datetime64("NaT")
    if value is _TIMEDELTA_NAT:
        return np.timedelta64("NaT")
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


def _compact_float32(values: np.ndarray) -> np.ndarray:
    """Use float32 only when every finite nonzero value remains usable."""

    array = np.asarray(values)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        compact = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(compact)):
        return array
    if np.any((array != 0.0) & (compact == 0.0)):
        return array
    return compact


def _install_roll_feature_row_precision(source_roll: Any) -> None:
    original = source_roll.roll_feature_row
    if getattr(original, _ROLL_PRECISION_MARKER, False):
        return

    @wraps(original)
    def roll_feature_row(
        row: Any,
        *,
        shift: Any,
        mode: str = "circular",
        fill_value: Any = 0.0,
    ) -> np.ndarray:
        vector = np.asarray(row, dtype=float).reshape(-1)
        if vector.shape[0] < 1:
            raise ValueError("row must contain at least one feature.")
        shift_value = source_roll._integer(shift, name="shift")
        normalized_mode = source_roll.normalize_roll_mode(mode)
        if normalized_mode == "circular":
            return _compact_float32(np.roll(vector, shift_value))
        fill = source_roll._finite_float(fill_value, name="fill_value")
        output = np.full(vector.shape[0], fill, dtype=float)
        if shift_value == 0:
            output[:] = vector
        elif abs(shift_value) < vector.shape[0]:
            if shift_value > 0:
                output[shift_value:] = vector[:-shift_value]
            else:
                output[:shift_value] = vector[-shift_value:]
        return _compact_float32(output)

    setattr(roll_feature_row, _ROLL_PRECISION_MARKER, True)
    source_roll.roll_feature_row = roll_feature_row


def install() -> None:
    """Install robust source feature-roll label and precision handling."""

    source_roll = importlib.import_module("neureptrace.decoding.source_roll")
    _install_roll_feature_row_precision(source_roll)

    original = source_roll.augment_source_with_feature_roll
    if getattr(original, _PATCH_MARKER, False):
        return

    def source_roll_label_vector(values: Any, *, expected_length: int, name: str) -> np.ndarray:
        return _label_vector(values, n_rows=expected_length, name=name)

    @wraps(original)
    def augment_source_with_feature_roll(
        source_features: Any,
        source_labels: Any,
        *,
        source_domains: Any = None,
        config: Any = None,
    ):
        cfg = source_roll.source_feature_roll_config() if config is None else source_roll._coerce_config(config)
        features = source_roll._feature_matrix(source_features, name="source_features")
        labels = source_roll_label_vector(source_labels, expected_length=features.shape[0], name="source_labels")
        domains = source_roll._domain_vector(source_domains, expected_length=features.shape[0])
        n_source_domains = source_roll._count_unique_hashable(domains)
        classes = np.asarray(tuple(dict.fromkeys(labels.tolist())), dtype=object)

        if not cfg.enabled:
            metadata = source_roll._metadata(
                cfg,
                n_source_rows=features.shape[0],
                n_synthetic_rows=0,
                n_classes=classes.shape[0],
                n_source_domains=n_source_domains,
                feature_dim=features.shape[1],
            )
            return source_roll.SourceFeatureRollResult(
                features=_compact_float32(features),
                labels=_object_vector(_restore_label(label) for label in labels.tolist()),
                synthetic_mask=np.zeros(features.shape[0], dtype=bool),
                content_indices=np.empty(0, dtype=int),
                shifts=np.empty(0, dtype=int),
                metadata=metadata,
            )

        rng = np.random.default_rng(cfg.random_state)
        synthetic_rows: list[np.ndarray] = []
        synthetic_labels: list[Any] = []
        content_indices: list[int] = []
        shifts: list[int] = []
        for class_label in classes.tolist():
            class_indices = np.flatnonzero(labels == class_label)
            if class_indices.size == 0:
                continue
            for _ in range(cfg.synthetic_per_class):
                content_index = int(rng.choice(class_indices))
                shift = source_roll.sample_roll_shift(
                    cfg.max_shift,
                    include_zero_shift=cfg.include_zero_shift,
                    rng=rng,
                )
                synthetic_rows.append(
                    source_roll.roll_feature_row(
                        features[content_index],
                        shift=shift,
                        mode=cfg.roll_mode,
                        fill_value=cfg.fill_value,
                    )
                )
                synthetic_labels.append(class_label)
                content_indices.append(content_index)
                shifts.append(shift)

        synthetic_features = (
            _compact_float32(np.vstack(synthetic_rows))
            if synthetic_rows
            else np.empty((0, features.shape[1]), dtype=np.float32)
        )
        synthetic_labels_array = np.asarray(synthetic_labels, dtype=object)
        if cfg.preserve_original:
            output_features = _compact_float32(np.vstack([features, synthetic_features]))
            output_labels = np.concatenate([labels, synthetic_labels_array])
            synthetic_mask = np.concatenate(
                [
                    np.zeros(features.shape[0], dtype=bool),
                    np.ones(synthetic_features.shape[0], dtype=bool),
                ]
            )
        else:
            output_features = synthetic_features
            output_labels = synthetic_labels_array
            synthetic_mask = np.ones(synthetic_features.shape[0], dtype=bool)

        metadata = source_roll._metadata(
            cfg,
            n_source_rows=features.shape[0],
            n_synthetic_rows=synthetic_features.shape[0],
            n_classes=classes.shape[0],
            n_source_domains=n_source_domains,
            feature_dim=features.shape[1],
        )
        return source_roll.SourceFeatureRollResult(
            features=output_features,
            labels=_object_vector(_restore_label(label) for label in output_labels.tolist()),
            synthetic_mask=synthetic_mask,
            content_indices=np.asarray(content_indices, dtype=int),
            shifts=np.asarray(shifts, dtype=int),
            metadata=metadata,
        )

    setattr(augment_source_with_feature_roll, _PATCH_MARKER, True)
    source_roll._label_vector = source_roll_label_vector
    source_roll.augment_source_with_feature_roll = augment_source_with_feature_roll


__all__ = ["install"]
