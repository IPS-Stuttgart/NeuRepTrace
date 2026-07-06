"""Strict source-only feature roll augmentation.

This module creates shifted copies of labeled source feature rows.  It is useful as
a small domain-generalization baseline when feature columns represent ordered
samples, time bins, or frequency bins.  Synthetic rows inherit the sampled source
label and are generated without using held-out data.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neureptrace._object_label_utils import label_counts, label_equal_mask

SOURCE_ROLL_AUGMENTATION = "source_feature_roll"
SOURCE_ROLL_PROTOCOL = "strict_source_only_feature_roll_augmentation"
SOURCE_ROLL_CATEGORY = "1_strict_source_only"
ROLL_MODES = ("circular", "constant")
DEFAULT_MAX_SHIFT = 3
_NONE_STRINGS = {"", "none", "null"}


@dataclass(frozen=True, slots=True)
class SourceFeatureRollConfig:
    """Configuration for source-only feature-roll augmentation."""

    synthetic_per_class: int = 0
    max_shift: int = DEFAULT_MAX_SHIFT
    roll_mode: str = "circular"
    fill_value: float = 0.0
    include_zero_shift: bool = False
    preserve_original: bool = True
    random_state: int | None = 13

    def __post_init__(self) -> None:
        """Normalize and validate direct dataclass construction."""

        object.__setattr__(
            self,
            "synthetic_per_class",
            _nonnegative_int(self.synthetic_per_class, name="synthetic_per_class"),
        )
        object.__setattr__(self, "max_shift", _positive_int(self.max_shift, name="max_shift"))
        object.__setattr__(self, "roll_mode", normalize_roll_mode(self.roll_mode))
        object.__setattr__(self, "fill_value", _finite_float(self.fill_value, name="fill_value"))
        object.__setattr__(
            self,
            "include_zero_shift",
            _bool_config(self.include_zero_shift, name="include_zero_shift"),
        )
        object.__setattr__(
            self,
            "preserve_original",
            _bool_config(self.preserve_original, name="preserve_original"),
        )
        object.__setattr__(self, "random_state", _optional_random_state(self.random_state))

    @property
    def enabled(self) -> bool:
        """Whether synthetic rows should be generated."""

        return self.synthetic_per_class > 0 and self.max_shift > 0


@dataclass(frozen=True, slots=True)
class SourceFeatureRollResult:
    """Augmented feature rows and provenance."""

    features: np.ndarray
    labels: np.ndarray
    synthetic_mask: np.ndarray
    content_indices: np.ndarray
    shifts: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_synthetic(self) -> int:
        """Number of generated rows in the returned matrix."""

        return int(np.sum(self.synthetic_mask))


# pylint: disable-next=too-many-locals
def augment_source_with_feature_roll(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    *,
    source_domains: Sequence[Hashable] | np.ndarray | None = None,
    config: SourceFeatureRollConfig | Mapping[str, Any] | None = None,
) -> SourceFeatureRollResult:
    """Append shifted source-row copies.

    The API intentionally has no held-out feature or held-out label arguments.
    """

    cfg = source_feature_roll_config() if config is None else _coerce_config(config)
    features = _feature_matrix(source_features, name="source_features")
    labels = _label_vector(source_labels, expected_length=features.shape[0], name="source_labels")
    domains = _domain_vector(source_domains, expected_length=features.shape[0])
    classes, _class_counts = label_counts(labels)
    domain_ids, _domain_counts = label_counts(domains)
    n_source_domains = int(domain_ids.shape[0])

    if not cfg.enabled:
        metadata = _metadata(
            cfg,
            n_source_rows=features.shape[0],
            n_synthetic_rows=0,
            n_classes=classes.shape[0],
            n_source_domains=n_source_domains,
            feature_dim=features.shape[1],
        )
        return SourceFeatureRollResult(
            features=features.astype(np.float32, copy=False),
            labels=labels.copy(),
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
        class_indices = np.flatnonzero(label_equal_mask(labels, class_label))
        if class_indices.size == 0:
            continue
        for _ in range(cfg.synthetic_per_class):
            content_index = int(rng.choice(class_indices))
            shift = sample_roll_shift(cfg.max_shift, include_zero_shift=cfg.include_zero_shift, rng=rng)
            synthetic_rows.append(roll_feature_row(features[content_index], shift=shift, mode=cfg.roll_mode, fill_value=cfg.fill_value))
            synthetic_labels.append(class_label)
            content_indices.append(content_index)
            shifts.append(shift)

    synthetic_features = np.vstack(synthetic_rows).astype(np.float32, copy=False) if synthetic_rows else np.empty((0, features.shape[1]), dtype=np.float32)
    synthetic_labels_array = _label_output_vector(synthetic_labels, dtype=labels.dtype)
    if cfg.preserve_original:
        output_features = np.vstack([features, synthetic_features]).astype(np.float32, copy=False)
        output_labels = np.concatenate([labels, synthetic_labels_array])
        synthetic_mask = np.concatenate([np.zeros(features.shape[0], dtype=bool), np.ones(synthetic_features.shape[0], dtype=bool)])
    else:
        output_features = synthetic_features
        output_labels = synthetic_labels_array
        synthetic_mask = np.ones(synthetic_features.shape[0], dtype=bool)

    metadata = _metadata(
        cfg,
        n_source_rows=features.shape[0],
        n_synthetic_rows=synthetic_features.shape[0],
        n_classes=classes.shape[0],
        n_source_domains=n_source_domains,
        feature_dim=features.shape[1],
    )
    return SourceFeatureRollResult(
        features=output_features,
        labels=output_labels,
        synthetic_mask=synthetic_mask,
        content_indices=np.asarray(content_indices, dtype=int),
        shifts=np.asarray(shifts, dtype=int),
        metadata=metadata,
    )


def roll_feature_row(
    row: Sequence[float] | np.ndarray,
    *,
    shift: int | str,
    mode: str = "circular",
    fill_value: float | str = 0.0,
) -> np.ndarray:
    """Roll one feature row by ``shift`` columns."""

    vector = np.asarray(row, dtype=float).reshape(-1)
    if vector.shape[0] < 1:
        raise ValueError("row must contain at least one feature.")
    shift_value = _integer(shift, name="shift")
    normalized_mode = normalize_roll_mode(mode)
    if normalized_mode == "circular":
        return np.roll(vector, shift_value).astype(np.float32, copy=False)
    fill = _finite_float(fill_value, name="fill_value")
    output = np.full(vector.shape[0], fill, dtype=float)
    if shift_value == 0:
        output[:] = vector
    elif abs(shift_value) < vector.shape[0]:
        if shift_value > 0:
            output[shift_value:] = vector[:-shift_value]
        else:
            output[:shift_value] = vector[-shift_value:]
    return output.astype(np.float32, copy=False)


def sample_roll_shift(max_shift: int | str, *, include_zero_shift: bool = False, rng: np.random.Generator | None = None) -> int:
    """Sample an integer shift in ``[-max_shift, max_shift]``."""

    maximum = _positive_int(max_shift, name="max_shift")
    generator = np.random.default_rng() if rng is None else rng
    candidates = np.arange(-maximum, maximum + 1, dtype=int)
    if not include_zero_shift:
        candidates = candidates[candidates != 0]
    return int(generator.choice(candidates))


def source_feature_roll_config(
    *,
    synthetic_per_class: int | str = 0,
    max_shift: int | str = DEFAULT_MAX_SHIFT,
    roll_mode: str | None = "circular",
    fill_value: float | str = 0.0,
    include_zero_shift: bool | str | int | float = False,
    preserve_original: bool | str | int | float = True,
    random_state: int | str | None = 13,
) -> SourceFeatureRollConfig:
    """Normalize public feature-roll options."""

    return SourceFeatureRollConfig(
        synthetic_per_class=synthetic_per_class,
        max_shift=max_shift,
        roll_mode=roll_mode,
        fill_value=fill_value,
        include_zero_shift=include_zero_shift,
        preserve_original=preserve_original,
        random_state=random_state,
    )


def normalize_roll_mode(value: Any) -> str:
    """Normalize roll-mode aliases."""

    value = _scalar_value(value, name="roll_mode")
    normalized = "circular" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"roll": "circular", "wrap": "circular", "zero": "constant", "pad": "constant", "zero_pad": "constant"}.get(normalized, normalized)
    if normalized not in ROLL_MODES:
        raise ValueError(f"Unknown roll_mode {value!r}. Available modes: {', '.join(ROLL_MODES)}.")
    return normalized


def _coerce_config(config: SourceFeatureRollConfig | Mapping[str, Any]) -> SourceFeatureRollConfig:
    if isinstance(config, SourceFeatureRollConfig):
        return config
    return source_feature_roll_config(**dict(config))


def _metadata(cfg: SourceFeatureRollConfig, *, n_source_rows: int, n_synthetic_rows: int, n_classes: int, n_source_domains: int, feature_dim: int) -> dict[str, Any]:
    return {
        "source_feature_roll": bool(cfg.enabled),
        "source_feature_roll_protocol": SOURCE_ROLL_PROTOCOL,
        "source_feature_roll_protocol_category": SOURCE_ROLL_CATEGORY,
        "source_feature_roll_method": SOURCE_ROLL_AUGMENTATION,
        "source_feature_roll_uses_source_features": True,
        "source_feature_roll_uses_source_labels": True,
        "source_feature_roll_uses_source_domains": True,
        "source_feature_roll_uses_heldout_features": False,
        "source_feature_roll_uses_heldout_labels": False,
        "source_feature_roll_valid_for_strict_source_only": True,
        "source_feature_roll_valid_for_benchmark": True,
        "source_feature_roll_n_source_rows": int(n_source_rows),
        "source_feature_roll_n_synthetic_rows": int(n_synthetic_rows),
        "source_feature_roll_n_output_rows": int(n_source_rows + n_synthetic_rows if cfg.preserve_original else n_synthetic_rows),
        "source_feature_roll_n_classes": int(n_classes),
        "source_feature_roll_n_source_domains": int(n_source_domains),
        "source_feature_roll_feature_dim": int(feature_dim),
        "source_feature_roll_synthetic_per_class": int(cfg.synthetic_per_class),
        "source_feature_roll_max_shift": int(cfg.max_shift),
        "source_feature_roll_mode": cfg.roll_mode,
        "source_feature_roll_fill_value": float(cfg.fill_value),
        "source_feature_roll_include_zero_shift": bool(cfg.include_zero_shift),
        "source_feature_roll_preserve_original": bool(cfg.preserve_original),
        "source_feature_roll_random_state": "" if cfg.random_state is None else int(cfg.random_state),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    return _atomic_value_vector(values, expected_length=expected_length, name=name, require_hashable=False)


def _domain_vector(values: Sequence[Hashable] | np.ndarray | None, *, expected_length: int) -> np.ndarray:
    if values is None:
        return np.full(expected_length, "source", dtype=object)
    return _atomic_value_vector(values, expected_length=expected_length, name="source_domains", require_hashable=True)


def _atomic_value_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str, require_hashable: bool) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        items = [values]
    else:
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            items = [array.item()]
        elif array.ndim == 1:
            if array.shape[0] == expected_length:
                items = array.tolist()
            elif expected_length == 1:
                items = [tuple(array.tolist())]
            else:
                items = array.reshape(-1).tolist()
        else:
            rows = array.reshape(array.shape[0], -1)
            if rows.shape[1] == 1:
                items = rows[:, 0].tolist()
            else:
                items = [tuple(row.tolist()) for row in rows]
    if len(items) != expected_length:
        raise ValueError(f"{name} must contain one value per feature row: {len(items)} != {expected_length}.")
    if require_hashable:
        for value in items:
            try:
                hash(value)
            except TypeError as exc:
                raise ValueError(f"{name} must be hashable; got {value!r}.") from exc
    if _contains_composite_value(items):
        return _object_vector(items)
    return np.asarray(items).reshape(-1)


def _label_output_vector(values: Sequence[Any], *, dtype: np.dtype) -> np.ndarray:
    if _contains_composite_value(values):
        return _object_vector(values)
    return np.asarray(values, dtype=dtype if dtype != object else object)


def _object_vector(values: Iterable[Any]) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _contains_composite_value(values: Sequence[Any]) -> bool:
    return any(_is_composite_value(value) for value in values)


def _is_composite_value(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim != 0
    return isinstance(value, (tuple, list, dict))


def _scalar_value(value: Any, *, name: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar value.")
        return value.item()
    return value


def _integer(value: int | str, *, name: str) -> int:
    value = _scalar_value(value, name=name)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    return int(parsed)


def _positive_int(value: int | str, *, name: str) -> int:
    parsed = _integer(value, name=name)
    if parsed < 1:
        raise ValueError(f"{name} must be positive.")
    return parsed


def _nonnegative_int(value: int | str, *, name: str) -> int:
    parsed = _integer(value, name=name)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative.")
    return parsed


def _finite_float(value: Any, *, name: str) -> float:
    value = _scalar_value(value, name=name)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed


def _optional_random_state(value: Any) -> int | None:
    value = _scalar_value(value, name="random_state")
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in _NONE_STRINGS:
        return None
    return _nonnegative_int(value, name="random_state")


def _bool_config(value: bool | str | int | float, *, name: str) -> bool:
    value = _scalar_value(value, name=name)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off"}:
            return False
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(float(value)) and float(value) in {0.0, 1.0}:
        return bool(value)
    raise ValueError(f"{name} must be a boolean value.")