"""Strict source-only feature masking augmentation.

The utilities in this module create masked copies of labeled source feature rows.
They are intended as a simple domain-generalization baseline for M/EEG feature
matrices: rows keep their source labels while a random feature subset or a
contiguous feature block is replaced by source-only fill statistics.

This is a Protocol-1 helper.  The public API uses source features and source
labels only, plus optional source-domain ids for provenance; held-out target data
are not accepted.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_MASKING_AUGMENTATION = "source_feature_masking"
SOURCE_MASKING_PROTOCOL = "strict_source_only_feature_masking_augmentation"
SOURCE_MASKING_CATEGORY = "1_strict_source_only"
MASK_MODES = ("feature", "block")
FILL_MODES = ("zero", "feature_mean", "row_mean")
DEFAULT_MASK_FRACTION = 0.15


@dataclass(frozen=True, slots=True)
class SourceFeatureMaskingConfig:
    """Configuration for strict source-only feature masking."""

    synthetic_per_class: int = 0
    mask_fraction: float = DEFAULT_MASK_FRACTION
    mask_mode: str = "feature"
    block_size: int | None = None
    fill_mode: str = "feature_mean"
    noise_std: float = 0.0
    preserve_original: bool = True
    random_state: int | None = 13

    @property
    def enabled(self) -> bool:
        """Whether synthetic rows should be generated."""

        return self.synthetic_per_class > 0 and self.mask_fraction > 0.0


@dataclass(frozen=True, slots=True)
class SourceFeatureMaskingResult:
    """Augmented source rows and provenance."""

    features: np.ndarray
    labels: np.ndarray
    synthetic_mask: np.ndarray
    content_indices: np.ndarray
    masked_feature_indices: tuple[np.ndarray, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_synthetic(self) -> int:
        """Number of generated rows in the returned matrix."""

        return int(np.sum(self.synthetic_mask))


# pylint: disable-next=too-many-locals

def augment_source_with_feature_masking(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    *,
    source_domains: Sequence[Hashable] | np.ndarray | None = None,
    config: SourceFeatureMaskingConfig | Mapping[str, Any] | None = None,
) -> SourceFeatureMaskingResult:
    """Append masked source-row copies.

    Parameters
    ----------
    source_features:
        Source feature matrix with rows as trials/windows and columns as features.
    source_labels:
        One source label per row.  Synthetic rows inherit the sampled row label.
    source_domains:
        Optional source-domain ids recorded only for provenance.
    config:
        Masking options.  Mappings are normalized through
        :func:`source_feature_masking_config`.

    Returns
    -------
    SourceFeatureMaskingResult
        Augmented feature rows, labels, synthetic-row mask, sampled content row ids,
        per-synthetic-row masked feature indices, and protocol metadata.
    """

    cfg = source_feature_masking_config() if config is None else _coerce_config(config)
    features = _feature_matrix(source_features, name="source_features")
    labels = _label_vector(source_labels, expected_length=features.shape[0], name="source_labels")
    domains = _domain_vector(source_domains, expected_length=features.shape[0])
    feature_fill = _feature_fill_values(features, mode=cfg.fill_mode)

    if not cfg.enabled:
        metadata = _metadata(cfg, n_source_rows=features.shape[0], n_synthetic_rows=0, n_classes=np.unique(labels).shape[0], n_source_domains=np.unique(domains).shape[0], feature_dim=features.shape[1])
        return SourceFeatureMaskingResult(
            features=features.astype(np.float32, copy=False),
            labels=labels.copy(),
            synthetic_mask=np.zeros(features.shape[0], dtype=bool),
            content_indices=np.empty(0, dtype=int),
            masked_feature_indices=(),
            metadata=metadata,
        )

    rng = np.random.default_rng(cfg.random_state)
    synthetic_rows: list[np.ndarray] = []
    synthetic_labels: list[Any] = []
    content_indices: list[int] = []
    masked_indices: list[np.ndarray] = []
    for class_label in tuple(dict.fromkeys(labels.tolist())):
        class_indices = np.flatnonzero(labels == class_label)
        if class_indices.size == 0:
            continue
        for _ in range(cfg.synthetic_per_class):
            content_index = int(rng.choice(class_indices))
            mask = feature_mask_indices(features.shape[1], mask_fraction=cfg.mask_fraction, mask_mode=cfg.mask_mode, block_size=cfg.block_size, rng=rng)
            row = features[content_index].copy()
            if cfg.fill_mode == "row_mean":
                row_fill = np.full(mask.shape[0], float(np.mean(features[content_index])), dtype=float)
            else:
                row_fill = feature_fill[mask]
            row[mask] = row_fill
            if cfg.noise_std > 0.0:
                row = row + rng.normal(0.0, cfg.noise_std, size=row.shape[0])
            synthetic_rows.append(row)
            synthetic_labels.append(class_label)
            content_indices.append(content_index)
            masked_indices.append(mask.astype(int, copy=False))

    synthetic_features = np.vstack(synthetic_rows).astype(np.float32, copy=False) if synthetic_rows else np.empty((0, features.shape[1]), dtype=np.float32)
    synthetic_labels_array = np.asarray(synthetic_labels, dtype=labels.dtype if labels.dtype != object else object)
    if cfg.preserve_original:
        output_features = np.vstack([features, synthetic_features]).astype(np.float32, copy=False)
        output_labels = np.concatenate([labels, synthetic_labels_array])
        synthetic_mask = np.concatenate([np.zeros(features.shape[0], dtype=bool), np.ones(synthetic_features.shape[0], dtype=bool)])
    else:
        output_features = synthetic_features
        output_labels = synthetic_labels_array
        synthetic_mask = np.ones(synthetic_features.shape[0], dtype=bool)

    metadata = _metadata(cfg, n_source_rows=features.shape[0], n_synthetic_rows=synthetic_features.shape[0], n_classes=np.unique(labels).shape[0], n_source_domains=np.unique(domains).shape[0], feature_dim=features.shape[1])
    return SourceFeatureMaskingResult(
        features=output_features,
        labels=output_labels,
        synthetic_mask=synthetic_mask,
        content_indices=np.asarray(content_indices, dtype=int),
        masked_feature_indices=tuple(masked_indices),
        metadata=metadata,
    )


def feature_mask_indices(
    n_features: int | str,
    *,
    mask_fraction: float | str = DEFAULT_MASK_FRACTION,
    mask_mode: str = "feature",
    block_size: int | str | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return sorted feature indices to replace for one synthetic row."""

    width = _positive_int(n_features, name="n_features")
    fraction = _unit_interval_float(mask_fraction, name="mask_fraction")
    mode = normalize_mask_mode(mask_mode)
    generator = np.random.default_rng() if rng is None else rng
    n_mask = max(1, int(round(width * fraction))) if fraction > 0.0 else 0
    n_mask = min(n_mask, width)
    if n_mask == 0:
        return np.empty(0, dtype=int)
    if mode == "feature":
        return np.sort(generator.choice(width, size=n_mask, replace=False)).astype(int, copy=False)
    size = n_mask if block_size is None else min(_positive_int(block_size, name="block_size"), width)
    start = int(generator.integers(0, width - size + 1))
    return np.arange(start, start + size, dtype=int)


def source_feature_masking_config(
    *,
    synthetic_per_class: int | str = 0,
    mask_fraction: float | str = DEFAULT_MASK_FRACTION,
    mask_mode: str | None = "feature",
    block_size: int | str | None = None,
    fill_mode: str | None = "feature_mean",
    noise_std: float | str = 0.0,
    preserve_original: bool = True,
    random_state: int | str | None = 13,
) -> SourceFeatureMaskingConfig:
    """Normalize public feature-masking options."""

    return SourceFeatureMaskingConfig(
        synthetic_per_class=_nonnegative_int(synthetic_per_class, name="synthetic_per_class"),
        mask_fraction=_unit_interval_float(mask_fraction, name="mask_fraction"),
        mask_mode=normalize_mask_mode(mask_mode),
        block_size=None if block_size in {None, "", "none", "None"} else _positive_int(block_size, name="block_size"),
        fill_mode=normalize_fill_mode(fill_mode),
        noise_std=_nonnegative_float(noise_std, name="noise_std"),
        preserve_original=bool(preserve_original),
        random_state=None if random_state in {None, "", "none", "None"} else _nonnegative_int(random_state, name="random_state"),
    )


def normalize_mask_mode(value: str | None) -> str:
    """Normalize mask-mode aliases."""

    normalized = "feature" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"random": "feature", "random_features": "feature", "contiguous": "block", "window": "block"}.get(normalized, normalized)
    if normalized not in MASK_MODES:
        raise ValueError(f"Unknown mask_mode {value!r}. Available modes: {', '.join(MASK_MODES)}.")
    return normalized


def normalize_fill_mode(value: str | None) -> str:
    """Normalize fill-mode aliases."""

    normalized = "feature_mean" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"mean": "feature_mean", "column_mean": "feature_mean", "trial_mean": "row_mean", "sample_mean": "row_mean"}.get(normalized, normalized)
    if normalized not in FILL_MODES:
        raise ValueError(f"Unknown fill_mode {value!r}. Available modes: {', '.join(FILL_MODES)}.")
    return normalized


def _coerce_config(config: SourceFeatureMaskingConfig | Mapping[str, Any]) -> SourceFeatureMaskingConfig:
    if isinstance(config, SourceFeatureMaskingConfig):
        return config
    return source_feature_masking_config(**dict(config))


def _feature_fill_values(features: np.ndarray, *, mode: str) -> np.ndarray:
    if mode == "zero":
        return np.zeros(features.shape[1], dtype=float)
    if mode == "feature_mean":
        return np.mean(features, axis=0)
    if mode == "row_mean":
        return np.zeros(features.shape[1], dtype=float)
    raise ValueError(f"Unhandled fill mode {mode!r}.")


def _metadata(cfg: SourceFeatureMaskingConfig, *, n_source_rows: int, n_synthetic_rows: int, n_classes: int, n_source_domains: int, feature_dim: int) -> dict[str, Any]:
    return {
        "source_feature_masking": bool(cfg.enabled),
        "source_feature_masking_protocol": SOURCE_MASKING_PROTOCOL,
        "source_feature_masking_protocol_category": SOURCE_MASKING_CATEGORY,
        "source_feature_masking_method": SOURCE_MASKING_AUGMENTATION,
        "source_feature_masking_uses_source_features": True,
        "source_feature_masking_uses_source_labels": True,
        "source_feature_masking_uses_source_domains": True,
        "source_feature_masking_uses_heldout_features": False,
        "source_feature_masking_uses_heldout_labels": False,
        "source_feature_masking_valid_for_strict_source_only": True,
        "source_feature_masking_valid_for_unlabeled_target_adaptation": True,
        "source_feature_masking_valid_for_benchmark": True,
        "source_feature_masking_n_source_rows": int(n_source_rows),
        "source_feature_masking_n_synthetic_rows": int(n_synthetic_rows),
        "source_feature_masking_n_output_rows": int(n_source_rows + n_synthetic_rows if cfg.preserve_original else n_synthetic_rows),
        "source_feature_masking_n_classes": int(n_classes),
        "source_feature_masking_n_source_domains": int(n_source_domains),
        "source_feature_masking_feature_dim": int(feature_dim),
        "source_feature_masking_synthetic_per_class": int(cfg.synthetic_per_class),
        "source_feature_masking_mask_fraction": float(cfg.mask_fraction),
        "source_feature_masking_mask_mode": cfg.mask_mode,
        "source_feature_masking_block_size": "" if cfg.block_size is None else int(cfg.block_size),
        "source_feature_masking_fill_mode": cfg.fill_mode,
        "source_feature_masking_noise_std": float(cfg.noise_std),
        "source_feature_masking_preserve_original": bool(cfg.preserve_original),
        "source_feature_masking_random_state": "" if cfg.random_state is None else int(cfg.random_state),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one feature column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = np.asarray(values).reshape(-1)
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one value per feature row: {vector.shape[0]} != {expected_length}.")
    return vector


def _domain_vector(values: Sequence[Hashable] | np.ndarray | None, *, expected_length: int) -> np.ndarray:
    if values is None:
        return np.full(expected_length, "source", dtype=object)
    vector = np.asarray(values, dtype=object).reshape(-1)
    if vector.shape[0] != expected_length:
        raise ValueError(f"source_domains must contain one value per feature row: {vector.shape[0]} != {expected_length}.")
    for value in vector.tolist():
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"source_domains must be hashable; got {value!r}.") from exc
    return vector


def _positive_int(value: int | str, *, name: str) -> int:
    integer = _integer(value, name=name)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _nonnegative_int(value: int | str, *, name: str) -> int:
    integer = _integer(value, name=name)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative.")
    return integer


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
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed
