"""Strict source-only SMOTE-style interpolation.

This module creates same-class interpolated source feature rows.  It is intended
as a dependency-light domain-generalization baseline for M/EEG feature matrices.
Synthetic rows keep the sampled source class label and are generated only from
source rows, so the protocol is strict source-only.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neureptrace._object_label_utils import label_counts, label_equal_mask

SOURCE_SMOTE_AUGMENTATION = "source_smote"
SOURCE_SMOTE_PROTOCOL = "strict_source_only_smote_interpolation"
SOURCE_SMOTE_CATEGORY = "1_strict_source_only"
_NONE_RANDOM_STATE_TOKENS = {"", "none", "null"}


@dataclass(frozen=True, slots=True)
class SourceSmoteConfig:
    """Configuration for source-only same-class interpolation."""

    synthetic_per_class: int = 0
    cross_domain_partner: bool = True
    preserve_original: bool = True
    random_state: int | None = 13
    jitter_std: float = 0.0

    def __post_init__(self) -> None:
        """Validate direct dataclass construction as strictly as the public helper."""

        object.__setattr__(self, "synthetic_per_class", _nonnegative_int(self.synthetic_per_class, name="synthetic_per_class"))
        object.__setattr__(self, "cross_domain_partner", _bool_value(self.cross_domain_partner, name="cross_domain_partner"))
        object.__setattr__(self, "preserve_original", _bool_value(self.preserve_original, name="preserve_original"))
        object.__setattr__(self, "random_state", _normalize_optional_random_state(self.random_state, name="random_state"))
        object.__setattr__(self, "jitter_std", _nonnegative_float(self.jitter_std, name="jitter_std"))

    @property
    def enabled(self) -> bool:
        """Whether this config requests synthetic rows."""

        return self.synthetic_per_class > 0


@dataclass(frozen=True, slots=True)
class SourceSmoteResult:
    """Augmented source rows and interpolation provenance."""

    features: np.ndarray
    labels: np.ndarray
    synthetic_mask: np.ndarray
    content_indices: np.ndarray
    partner_indices: np.ndarray
    lambdas: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_synthetic(self) -> int:
        """Number of generated rows in the returned matrix."""

        return int(np.sum(self.synthetic_mask))


# pylint: disable-next=too-many-locals
def augment_source_with_smote(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    *,
    source_domains: Sequence[Hashable] | np.ndarray | None = None,
    config: SourceSmoteConfig | Mapping[str, Any] | None = None,
) -> SourceSmoteResult:
    """Append same-class source interpolation rows.

    Parameters
    ----------
    source_features:
        Source feature matrix.
    source_labels:
        One source label per row.  Synthetic rows inherit the sampled class label.
    source_domains:
        Optional source-domain identifiers.  When ``cross_domain_partner=True``,
        interpolation partners are drawn from another source domain when possible.
    config:
        SMOTE options.  Mappings are normalized through :func:`source_smote_config`.
    """

    cfg = source_smote_config() if config is None else _coerce_config(config)
    features = _feature_matrix(source_features, name="source_features")
    labels = _label_vector(source_labels, expected_length=features.shape[0], name="source_labels")
    domains = _domain_vector(source_domains, expected_length=features.shape[0])
    classes, _class_counts = label_counts(labels)
    domain_ids, _domain_counts = label_counts(domains)

    if not cfg.enabled:
        metadata = _metadata(
            cfg,
            n_source_rows=features.shape[0],
            n_synthetic_rows=0,
            n_classes=classes.shape[0],
            n_source_domains=domain_ids.shape[0],
            feature_dim=features.shape[1],
        )
        return SourceSmoteResult(
            features=features.astype(np.float32, copy=False),
            labels=labels.copy(),
            synthetic_mask=np.zeros(features.shape[0], dtype=bool),
            content_indices=np.empty(0, dtype=int),
            partner_indices=np.empty(0, dtype=int),
            lambdas=np.empty(0, dtype=float),
            metadata=metadata,
        )

    rng = np.random.default_rng(cfg.random_state)
    synthetic_rows: list[np.ndarray] = []
    synthetic_labels: list[Any] = []
    content_indices: list[int] = []
    partner_indices: list[int] = []
    lambdas: list[float] = []

    for class_label in classes.tolist():
        class_indices = np.flatnonzero(label_equal_mask(labels, class_label))
        if class_indices.size == 0:
            continue
        for _ in range(cfg.synthetic_per_class):
            content_index = int(rng.choice(class_indices))
            partner_pool = class_indices[class_indices != content_index] if class_indices.size > 1 else class_indices
            if cfg.cross_domain_partner and partner_pool.size > 0:
                same_domain_mask = label_equal_mask(domains[partner_pool], domains[content_index])
                cross_pool = partner_pool[~same_domain_mask]
                if cross_pool.size:
                    partner_pool = cross_pool
            if partner_pool.size == 0:
                partner_pool = np.asarray([content_index], dtype=int)
            partner_index = int(rng.choice(partner_pool))
            lam = float(rng.random())
            row = interpolate_rows(features[content_index], features[partner_index], lam)
            if cfg.jitter_std > 0.0:
                row = row + rng.normal(0.0, cfg.jitter_std, size=row.shape[0])
            synthetic_rows.append(row)
            synthetic_labels.append(class_label)
            content_indices.append(content_index)
            partner_indices.append(partner_index)
            lambdas.append(lam)

    synthetic_features = np.vstack(synthetic_rows).astype(np.float32, copy=False) if synthetic_rows else np.empty((0, features.shape[1]), dtype=np.float32)
    synthetic_label_array = _object_value_vector(synthetic_labels)
    if cfg.preserve_original:
        output_features = np.vstack([features, synthetic_features]).astype(np.float32, copy=False)
        output_labels = np.concatenate([labels, synthetic_label_array])
        synthetic_mask = np.concatenate([np.zeros(features.shape[0], dtype=bool), np.ones(synthetic_features.shape[0], dtype=bool)])
    else:
        output_features = synthetic_features
        output_labels = synthetic_label_array
        synthetic_mask = np.ones(synthetic_features.shape[0], dtype=bool)

    metadata = _metadata(
        cfg,
        n_source_rows=features.shape[0],
        n_synthetic_rows=synthetic_features.shape[0],
        n_classes=classes.shape[0],
        n_source_domains=domain_ids.shape[0],
        feature_dim=features.shape[1],
    )
    return SourceSmoteResult(
        features=output_features,
        labels=output_labels,
        synthetic_mask=synthetic_mask,
        content_indices=np.asarray(content_indices, dtype=int),
        partner_indices=np.asarray(partner_indices, dtype=int),
        lambdas=np.asarray(lambdas, dtype=float),
        metadata=metadata,
    )


def interpolate_rows(content_row: Sequence[float] | np.ndarray, partner_row: Sequence[float] | np.ndarray, lam: float | str) -> np.ndarray:
    """Return a convex interpolation of two feature rows."""

    left = np.asarray(content_row, dtype=float).reshape(-1)
    right = np.asarray(partner_row, dtype=float).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("content_row and partner_row must be non-empty vectors with the same shape.")
    weight = _unit_interval_float(lam, name="lam")
    return (left + weight * (right - left)).astype(np.float32, copy=False)


def source_smote_config(
    *,
    synthetic_per_class: int | str = 0,
    cross_domain_partner: bool | int | str = True,
    preserve_original: bool | int | str = True,
    random_state: Any = 13,
    jitter_std: float | str = 0.0,
) -> SourceSmoteConfig:
    """Normalize public source-SMOTE options."""

    return SourceSmoteConfig(
        synthetic_per_class=_nonnegative_int(synthetic_per_class, name="synthetic_per_class"),
        cross_domain_partner=_bool_value(cross_domain_partner, name="cross_domain_partner"),
        preserve_original=_bool_value(preserve_original, name="preserve_original"),
        random_state=_normalize_optional_random_state(random_state, name="random_state"),
        jitter_std=_nonnegative_float(jitter_std, name="jitter_std"),
    )


def _random_state_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a non-negative integer or none.")


def _is_none_like_random_state(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _NONE_RANDOM_STATE_TOKENS
    return False


def _normalize_optional_random_state(value: Any, *, name: str) -> int | None:
    if _is_none_like_random_state(value):
        return None
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _random_state_error(name)
        value = value.item()
        if _is_none_like_random_state(value):
            return None
    if isinstance(value, (list, tuple, dict, set)):
        raise _random_state_error(name)
    try:
        return _nonnegative_int(value, name=name)
    except ValueError as exc:
        raise _random_state_error(name) from exc


def _coerce_config(config: SourceSmoteConfig | Mapping[str, Any]) -> SourceSmoteConfig:
    if isinstance(config, SourceSmoteConfig):
        return source_smote_config(
            synthetic_per_class=config.synthetic_per_class,
            cross_domain_partner=config.cross_domain_partner,
            preserve_original=config.preserve_original,
            random_state=config.random_state,
            jitter_std=config.jitter_std,
        )
    return source_smote_config(**dict(config))


def _metadata(cfg: SourceSmoteConfig, *, n_source_rows: int, n_synthetic_rows: int, n_classes: int, n_source_domains: int, feature_dim: int) -> dict[str, Any]:
    return {
        "source_smote": bool(cfg.enabled),
        "source_smote_protocol": SOURCE_SMOTE_PROTOCOL,
        "source_smote_protocol_category": SOURCE_SMOTE_CATEGORY,
        "source_smote_method": SOURCE_SMOTE_AUGMENTATION,
        "source_smote_uses_source_features": True,
        "source_smote_uses_source_labels": True,
        "source_smote_uses_source_domains": True,
        "source_smote_uses_heldout_features": False,
        "source_smote_uses_heldout_labels": False,
        "source_smote_valid_for_strict_source_only": True,
        "source_smote_valid_for_unlabeled_target_adaptation": True,
        "source_smote_valid_for_benchmark": True,
        "source_smote_n_source_rows": int(n_source_rows),
        "source_smote_n_synthetic_rows": int(n_synthetic_rows),
        "source_smote_n_output_rows": int(n_source_rows + n_synthetic_rows if cfg.preserve_original else n_synthetic_rows),
        "source_smote_n_classes": int(n_classes),
        "source_smote_n_source_domains": int(n_source_domains),
        "source_smote_feature_dim": int(feature_dim),
        "source_smote_synthetic_per_class": int(cfg.synthetic_per_class),
        "source_smote_cross_domain_partner": bool(cfg.cross_domain_partner),
        "source_smote_preserve_original": bool(cfg.preserve_original),
        "source_smote_random_state": "" if cfg.random_state is None else int(cfg.random_state),
        "source_smote_jitter_std": float(cfg.jitter_std),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _object_value_vector(values: Iterable[Any]) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _atomic_value_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        vector = _object_value_vector([values])
    else:
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            vector = _object_value_vector([array.item()])
        elif array.ndim == 1:
            if array.shape[0] == expected_length:
                vector = array.reshape(-1)
            elif expected_length == 1:
                vector = _object_value_vector([tuple(array.tolist())])
            else:
                vector = array.reshape(-1)
        else:
            rows = array.reshape(array.shape[0], -1)
            if rows.shape[1] == 1:
                vector = rows[:, 0].reshape(-1)
            else:
                vector = _object_value_vector(tuple(row.tolist()) for row in rows)
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one value per feature row: {vector.shape[0]} != {expected_length}.")
    return vector


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    return _atomic_value_vector(values, expected_length=expected_length, name=name)


def _domain_vector(values: Sequence[Hashable] | np.ndarray | None, *, expected_length: int) -> np.ndarray:
    if values is None:
        return np.full(expected_length, "source", dtype=object)
    vector = _atomic_value_vector(values, expected_length=expected_length, name="source_domains")
    for value in vector.tolist():
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"source_domains must be hashable; got {value!r}.") from exc
    return vector


def _numeric_scalar_input(value: Any, *, message: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
    if isinstance(value, (list, tuple, dict, set)):
        raise ValueError(message)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    return value


def _nonnegative_int(value: int | str, *, name: str) -> int:
    integer = _integer(value, name=name)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative.")
    return integer


def _integer(value: int | str, *, name: str) -> int:
    message = f"{name} must be an integer."
    value = _numeric_scalar_input(value, message=message)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0:
        raise ValueError(message)
    return int(parsed)


def _unit_interval_float(value: float | str, *, name: str) -> float:
    parsed = _float_value(value, name=name)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed


def _nonnegative_float(value: float | str, *, name: str) -> float:
    parsed = _float_value(value, name=name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return parsed


def _float_value(value: float | str, *, name: str) -> float:
    message = f"{name} must be finite."
    value = _numeric_scalar_input(value, message=message)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(parsed):
        raise ValueError(message)
    return parsed


def _bool_value(value: bool | int | str, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        if int(value) in {0, 1}:
            return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean.")
