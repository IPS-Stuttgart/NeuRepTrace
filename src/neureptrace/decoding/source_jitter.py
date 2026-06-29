"""Strict source-only feature jitter augmentation.

This module creates Gaussian-perturbed copies of labeled source feature rows.  The
noise scale is estimated from source rows only, either globally or per class.  The
synthetic rows inherit the sampled source label and can be appended to a fold-local
training matrix before fitting an ordinary decoder.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neureptrace._object_label_utils import label_counts, label_equal_mask

SOURCE_JITTER_AUGMENTATION = "source_feature_jitter"
SOURCE_JITTER_PROTOCOL = "strict_source_only_feature_jitter_augmentation"
SOURCE_JITTER_CATEGORY = "1_strict_source_only"
SCALE_MODES = ("global", "class", "unit")
DEFAULT_NOISE_SCALE = 0.05
DEFAULT_EPSILON = 1e-8
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}
_NONE_STRINGS = {"", "none", "null"}


@dataclass(frozen=True, slots=True)
class SourceFeatureJitterConfig:
    """Configuration for strict source-only feature jitter."""

    synthetic_per_class: int = 0
    noise_scale: float = DEFAULT_NOISE_SCALE
    scale_mode: str = "global"
    preserve_original: bool = True
    random_state: int | None = 13
    epsilon: float = DEFAULT_EPSILON

    @property
    def enabled(self) -> bool:
        """Whether synthetic rows should be generated."""

        return self.synthetic_per_class > 0 and self.noise_scale > 0.0


@dataclass(frozen=True, slots=True)
class SourceFeatureJitterResult:
    """Augmented features, labels, and provenance metadata."""

    features: np.ndarray
    labels: np.ndarray
    synthetic_mask: np.ndarray
    content_indices: np.ndarray
    noise: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_synthetic(self) -> int:
        """Number of generated rows in the returned matrix."""

        return int(np.sum(self.synthetic_mask))


# pylint: disable-next=too-many-locals

def augment_source_with_feature_jitter(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    *,
    source_domains: Sequence[Hashable] | np.ndarray | None = None,
    config: SourceFeatureJitterConfig | Mapping[str, Any] | None = None,
) -> SourceFeatureJitterResult:
    """Append Gaussian-jittered source-row copies.

    Parameters
    ----------
    source_features:
        Source feature matrix with rows as trials/windows and columns as features.
    source_labels:
        One source label per row.  Synthetic rows inherit the sampled source row
        label.
    source_domains:
        Optional source-domain ids recorded only for provenance.
    config:
        Jitter settings.  Mappings are normalized through
        :func:`source_feature_jitter_config`.
    """

    cfg = source_feature_jitter_config() if config is None else _coerce_config(config)
    features = _feature_matrix(source_features, name="source_features")
    labels = _label_vector(source_labels, expected_length=features.shape[0], name="source_labels")
    domains = _domain_vector(source_domains, expected_length=features.shape[0])
    classes, _class_counts = label_counts(labels)
    n_source_domains = int(label_counts(domains)[0].shape[0])
    scale_by_class = _scale_by_class(features, labels, classes=classes, mode=cfg.scale_mode, epsilon=cfg.epsilon)

    if not cfg.enabled:
        metadata = _metadata(
            cfg,
            n_source_rows=features.shape[0],
            n_synthetic_rows=0,
            n_classes=classes.shape[0],
            n_source_domains=n_source_domains,
            feature_dim=features.shape[1],
        )
        return SourceFeatureJitterResult(
            features=features.astype(np.float32, copy=False),
            labels=labels.copy(),
            synthetic_mask=np.zeros(features.shape[0], dtype=bool),
            content_indices=np.empty(0, dtype=int),
            noise=np.empty((0, features.shape[1]), dtype=np.float32),
            metadata=metadata,
        )

    rng = np.random.default_rng(cfg.random_state)
    synthetic_rows: list[np.ndarray] = []
    synthetic_labels: list[Any] = []
    content_indices: list[int] = []
    noise_rows: list[np.ndarray] = []
    for class_position, class_label in enumerate(classes.tolist()):
        class_indices = np.flatnonzero(label_equal_mask(labels, class_label))
        if class_indices.size == 0:
            continue
        class_scale = scale_by_class[class_position]
        for _ in range(cfg.synthetic_per_class):
            row_index = int(rng.choice(class_indices))
            noise = rng.normal(0.0, cfg.noise_scale, size=features.shape[1]) * class_scale
            synthetic_rows.append(features[row_index] + noise)
            synthetic_labels.append(class_label)
            content_indices.append(row_index)
            noise_rows.append(noise)

    synthetic_features = np.vstack(synthetic_rows).astype(np.float32, copy=False) if synthetic_rows else np.empty((0, features.shape[1]), dtype=np.float32)
    synthetic_labels_array = _label_output_vector(synthetic_labels, dtype=labels.dtype)
    noise_matrix = np.vstack(noise_rows).astype(np.float32, copy=False) if noise_rows else np.empty((0, features.shape[1]), dtype=np.float32)
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
    return SourceFeatureJitterResult(
        features=output_features,
        labels=output_labels,
        synthetic_mask=synthetic_mask,
        content_indices=np.asarray(content_indices, dtype=int),
        noise=noise_matrix,
        metadata=metadata,
    )


def source_feature_jitter_config(
    *,
    synthetic_per_class: int | str = 0,
    noise_scale: float | str = DEFAULT_NOISE_SCALE,
    scale_mode: str | None = "global",
    preserve_original: bool | str = True,
    random_state: int | str | None = 13,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceFeatureJitterConfig:
    """Normalize public jitter options."""

    return SourceFeatureJitterConfig(
        synthetic_per_class=_nonnegative_int(synthetic_per_class, name="synthetic_per_class"),
        noise_scale=_nonnegative_float(noise_scale, name="noise_scale"),
        scale_mode=normalize_jitter_scale_mode(scale_mode),
        preserve_original=_boolean(preserve_original, name="preserve_original"),
        random_state=_nonnegative_optional_int(random_state, name="random_state"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def normalize_jitter_scale_mode(value: str | None) -> str:
    """Normalize noise-scale mode aliases."""

    normalized = "global" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"pooled": "global", "all": "global", "classwise": "class", "per_class": "class", "identity": "unit", "none": "unit"}.get(normalized, normalized)
    if normalized not in SCALE_MODES:
        raise ValueError(f"Unknown scale_mode {value!r}. Available modes: {', '.join(SCALE_MODES)}.")
    return normalized


def _coerce_config(config: SourceFeatureJitterConfig | Mapping[str, Any]) -> SourceFeatureJitterConfig:
    if isinstance(config, SourceFeatureJitterConfig):
        return config
    return source_feature_jitter_config(**dict(config))


def _scale_by_class(features: np.ndarray, labels: np.ndarray, *, classes: np.ndarray, mode: str, epsilon: float) -> np.ndarray:
    if mode == "unit":
        return np.ones((classes.shape[0], features.shape[1]), dtype=float)
    if mode == "global":
        global_scale = _feature_scale(features, epsilon=epsilon)
        return np.repeat(global_scale[None, :], classes.shape[0], axis=0)
    if mode == "class":
        rows = []
        for class_label in classes.tolist():
            class_rows = features[label_equal_mask(labels, class_label)]
            rows.append(_feature_scale(class_rows, epsilon=epsilon))
        return np.vstack(rows)
    raise ValueError(f"Unhandled scale mode {mode!r}.")


def _feature_scale(features: np.ndarray, *, epsilon: float) -> np.ndarray:
    ddof = 1 if features.shape[0] > 1 else 0
    scale = np.std(features - np.mean(features, axis=0), axis=0, ddof=ddof)
    return np.maximum(scale, float(epsilon))


def _metadata(cfg: SourceFeatureJitterConfig, *, n_source_rows: int, n_synthetic_rows: int, n_classes: int, n_source_domains: int, feature_dim: int) -> dict[str, Any]:
    return {
        "source_feature_jitter": bool(cfg.enabled),
        "source_feature_jitter_protocol": SOURCE_JITTER_PROTOCOL,
        "source_feature_jitter_protocol_category": SOURCE_JITTER_CATEGORY,
        "source_feature_jitter_method": SOURCE_JITTER_AUGMENTATION,
        "source_feature_jitter_uses_source_features": True,
        "source_feature_jitter_uses_source_labels": True,
        "source_feature_jitter_uses_source_domains": True,
        "source_feature_jitter_uses_heldout_features": False,
        "source_feature_jitter_uses_heldout_labels": False,
        "source_feature_jitter_valid_for_strict_source_only": True,
        "source_feature_jitter_valid_for_unlabeled_target_adaptation": True,
        "source_feature_jitter_valid_for_benchmark": True,
        "source_feature_jitter_n_source_rows": int(n_source_rows),
        "source_feature_jitter_n_synthetic_rows": int(n_synthetic_rows),
        "source_feature_jitter_n_output_rows": int(n_source_rows + n_synthetic_rows if cfg.preserve_original else n_synthetic_rows),
        "source_feature_jitter_n_classes": int(n_classes),
        "source_feature_jitter_n_source_domains": int(n_source_domains),
        "source_feature_jitter_feature_dim": int(feature_dim),
        "source_feature_jitter_synthetic_per_class": int(cfg.synthetic_per_class),
        "source_feature_jitter_noise_scale": float(cfg.noise_scale),
        "source_feature_jitter_scale_mode": cfg.scale_mode,
        "source_feature_jitter_preserve_original": bool(cfg.preserve_original),
        "source_feature_jitter_random_state": "" if cfg.random_state is None else int(cfg.random_state),
        "source_feature_jitter_epsilon": float(cfg.epsilon),
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


def _object_vector(values: Iterable[Any]) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _label_output_vector(values: Sequence[Any], *, dtype: np.dtype) -> np.ndarray:
    if _contains_composite_value(values):
        return _object_vector(values)
    return np.asarray(values, dtype=dtype if dtype is not object else object)


def _contains_composite_value(values: Sequence[Any]) -> bool:
    return any(_is_composite_value(value) for value in values)


def _is_composite_value(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim != 0
    return isinstance(value, (tuple, list, dict))


def _optional_integer(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in _NONE_STRINGS:
            return None
        value = text
    elif isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be an integer or None.")
        return _optional_integer(value.item(), name=name)
    elif isinstance(value, (list, tuple, dict, set)):
        raise ValueError(f"{name} must be an integer or None.")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer or None.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer or None.") from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer or None.")
    return int(parsed)


def _nonnegative_optional_int(value: Any, *, name: str) -> int | None:
    parsed = _optional_integer(value, name=name)
    if parsed is None:
        return None
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer or None.")
    return parsed


def _nonnegative_int(value: int | str, *, name: str) -> int:
    parsed = _integer(value, name=name)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative.")
    return parsed


def _integer(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    return int(parsed)


def _nonnegative_float(value: float | str, *, name: str) -> float:
    parsed = _float_value(value, name=name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return parsed


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = _float_value(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _float_value(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed


def _boolean(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
        raise ValueError(f"{name} must be a boolean value.")
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a boolean value.")
        return _boolean(value.item(), name=name)
    if isinstance(value, (int, np.integer)):
        if int(value) in {0, 1}:
            return bool(value)
        raise ValueError(f"{name} must be a boolean value.")
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value) in {0.0, 1.0}:
            return bool(value)
        raise ValueError(f"{name} must be a boolean value.")
    raise ValueError(f"{name} must be a boolean value.")
