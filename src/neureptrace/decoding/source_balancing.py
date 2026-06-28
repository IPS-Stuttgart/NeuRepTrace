"""Strict source-only class balancing for feature matrices.

This module provides dependency-light class balancing utilities for fold-local
cross-subject decoding.  It can oversample minority source classes, undersample
majority source classes, or return per-row sample weights.  The implementation
uses source features and source labels only, so it is a Protocol-1 helper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_BALANCING_PROTOCOL = "strict_source_only_class_balancing"
SOURCE_BALANCING_CATEGORY = "1_strict_source_only"
BALANCING_MODES = ("oversample", "undersample", "weights")
TARGET_COUNT_MODES = ("max", "min", "median", "mean")


@dataclass(frozen=True, slots=True)
class SourceClassBalancingConfig:
    """Configuration for source-only class balancing."""

    mode: str = "oversample"
    target_count: int | str = "max"
    random_state: int | None = 13
    preserve_order: bool = False


@dataclass(frozen=True, slots=True)
class SourceClassBalancingResult:
    """Balanced source rows, labels, weights, and provenance."""

    features: np.ndarray
    labels: np.ndarray
    selected_indices: np.ndarray
    sample_weight: np.ndarray
    synthetic_mask: np.ndarray
    classes: np.ndarray
    class_counts_before: Mapping[Any, int]
    class_counts_after: Mapping[Any, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        """Number of output rows."""

        return int(self.features.shape[0])


# pylint: disable-next=too-many-locals

def balance_source_classes(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    *,
    config: SourceClassBalancingConfig | Mapping[str, Any] | None = None,
) -> SourceClassBalancingResult:
    """Balance source rows by resampling or sample weighting.

    Parameters
    ----------
    source_features:
        Source feature matrix.
    source_labels:
        One source label per row.
    config:
        Balancing options.  Mappings are normalized through
        :func:`source_class_balancing_config`.

    Returns
    -------
    SourceClassBalancingResult
        Balanced features, labels, selected input indices, sample weights, and
        metadata.  In ``weights`` mode, rows are not resampled and
        ``sample_weight`` contains inverse-frequency weights.
    """

    cfg = source_class_balancing_config() if config is None else _coerce_config(config)
    features = _feature_matrix(source_features, name="source_features")
    labels = _label_vector(source_labels, expected_length=features.shape[0], name="source_labels")
    classes = np.asarray(tuple(dict.fromkeys(labels.tolist())), dtype=labels.dtype if labels.dtype != object else object)
    if classes.shape[0] < 1:
        raise ValueError("At least one class is required.")
    counts = {class_label: int(np.count_nonzero(labels == class_label)) for class_label in classes.tolist()}
    target_count = resolve_target_count(tuple(counts.values()), cfg.target_count)

    if cfg.mode == "weights":
        selected = np.arange(features.shape[0], dtype=int)
        weights = _inverse_frequency_weights(labels, counts)
        synthetic_mask = np.zeros(features.shape[0], dtype=bool)
    else:
        rng = np.random.default_rng(cfg.random_state)
        selected_parts: list[np.ndarray] = []
        synthetic_parts: list[np.ndarray] = []
        for class_label in classes.tolist():
            class_indices = np.flatnonzero(labels == class_label)
            if cfg.mode == "oversample":
                if class_indices.size >= target_count:
                    choice = class_indices.copy()
                    synthetic = np.zeros(choice.shape[0], dtype=bool)
                else:
                    extra = rng.choice(class_indices, size=target_count - class_indices.size, replace=True)
                    choice = np.concatenate([class_indices, extra])
                    synthetic = np.concatenate([np.zeros(class_indices.shape[0], dtype=bool), np.ones(extra.shape[0], dtype=bool)])
            elif cfg.mode == "undersample":
                size = min(class_indices.size, target_count)
                choice = rng.choice(class_indices, size=size, replace=False)
                synthetic = np.zeros(choice.shape[0], dtype=bool)
            else:  # pragma: no cover - guarded by config normalization
                raise ValueError(f"Unhandled balancing mode {cfg.mode!r}.")
            selected_parts.append(choice.astype(int, copy=False))
            synthetic_parts.append(synthetic)
        selected = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=int)
        synthetic_mask = np.concatenate(synthetic_parts) if synthetic_parts else np.empty(0, dtype=bool)
        if cfg.preserve_order:
            order = np.argsort(selected, kind="stable")
        else:
            order = rng.permutation(selected.shape[0])
        selected = selected[order]
        synthetic_mask = synthetic_mask[order]
        weights = np.ones(selected.shape[0], dtype=float)

    output_features = features[selected].astype(np.float32, copy=False)
    output_labels = labels[selected]
    after_counts = {class_label: int(np.count_nonzero(output_labels == class_label)) for class_label in classes.tolist()}
    metadata = _metadata(
        cfg,
        n_source_rows=features.shape[0],
        n_output_rows=output_features.shape[0],
        n_classes=classes.shape[0],
        feature_dim=features.shape[1],
        counts_before=counts,
        counts_after=after_counts,
        target_count=target_count,
        n_synthetic=int(np.count_nonzero(synthetic_mask)),
    )
    return SourceClassBalancingResult(
        features=output_features,
        labels=output_labels,
        selected_indices=selected,
        sample_weight=weights.astype(np.float32, copy=False),
        synthetic_mask=synthetic_mask,
        classes=classes,
        class_counts_before=counts,
        class_counts_after=after_counts,
        metadata=metadata,
    )


def source_class_balancing_config(
    *,
    mode: str | None = "oversample",
    target_count: int | str = "max",
    random_state: int | str | None = 13,
    preserve_order: bool = False,
) -> SourceClassBalancingConfig:
    """Normalize public class-balancing options."""

    return SourceClassBalancingConfig(
        mode=normalize_balancing_mode(mode),
        target_count=target_count,
        random_state=None if random_state in {None, "", "none", "None"} else _nonnegative_int(random_state, name="random_state"),
        preserve_order=bool(preserve_order),
    )


def normalize_balancing_mode(value: str | None) -> str:
    """Normalize balancing mode aliases."""

    normalized = "oversample" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {
        "over": "oversample",
        "up": "oversample",
        "upsample": "oversample",
        "under": "undersample",
        "down": "undersample",
        "downsample": "undersample",
        "weight": "weights",
        "sample_weight": "weights",
        "inverse_frequency": "weights",
    }.get(normalized, normalized)
    if normalized not in BALANCING_MODES:
        raise ValueError(f"Unknown balancing mode {value!r}. Available modes: {', '.join(BALANCING_MODES)}.")
    return normalized


def resolve_target_count(class_counts: Sequence[int], target_count: int | str) -> int:
    """Resolve a concrete target row count from class counts."""

    counts = np.asarray(class_counts, dtype=float).reshape(-1)
    if counts.size == 0 or np.any(counts < 1):
        raise ValueError("class_counts must contain positive counts.")
    if isinstance(target_count, str):
        normalized = target_count.strip().lower().replace("-", "_")
        if normalized == "max":
            return int(np.max(counts))
        if normalized == "min":
            return int(np.min(counts))
        if normalized == "median":
            return int(round(float(np.median(counts))))
        if normalized == "mean":
            return int(round(float(np.mean(counts))))
        return _positive_int(normalized, name="target_count")
    return _positive_int(target_count, name="target_count")


def _inverse_frequency_weights(labels: np.ndarray, counts: Mapping[Any, int]) -> np.ndarray:
    n_classes = len(counts)
    weights = np.empty(labels.shape[0], dtype=float)
    for class_label, count in counts.items():
        weights[labels == class_label] = labels.shape[0] / float(n_classes * count)
    return weights


def _coerce_config(config: SourceClassBalancingConfig | Mapping[str, Any]) -> SourceClassBalancingConfig:
    if isinstance(config, SourceClassBalancingConfig):
        return config
    return source_class_balancing_config(**dict(config))


def _metadata(
    cfg: SourceClassBalancingConfig,
    *,
    n_source_rows: int,
    n_output_rows: int,
    n_classes: int,
    feature_dim: int,
    counts_before: Mapping[Any, int],
    counts_after: Mapping[Any, int],
    target_count: int,
    n_synthetic: int,
) -> dict[str, Any]:
    return {
        "source_class_balancing": True,
        "source_class_balancing_protocol": SOURCE_BALANCING_PROTOCOL,
        "source_class_balancing_protocol_category": SOURCE_BALANCING_CATEGORY,
        "source_class_balancing_mode": cfg.mode,
        "source_class_balancing_uses_source_features": True,
        "source_class_balancing_uses_source_labels": True,
        "source_class_balancing_uses_heldout_features": False,
        "source_class_balancing_uses_heldout_labels": False,
        "source_class_balancing_valid_for_strict_source_only": True,
        "source_class_balancing_valid_for_benchmark": True,
        "source_class_balancing_n_source_rows": int(n_source_rows),
        "source_class_balancing_n_output_rows": int(n_output_rows),
        "source_class_balancing_n_synthetic_rows": int(n_synthetic),
        "source_class_balancing_n_classes": int(n_classes),
        "source_class_balancing_feature_dim": int(feature_dim),
        "source_class_balancing_target_count": int(target_count),
        "source_class_balancing_random_state": "" if cfg.random_state is None else int(cfg.random_state),
        "source_class_balancing_preserve_order": bool(cfg.preserve_order),
        "source_class_balancing_counts_before": _format_counts(counts_before),
        "source_class_balancing_counts_after": _format_counts(counts_after),
    }


def _format_counts(counts: Mapping[Any, int]) -> str:
    return "|".join(f"{label}:{int(count)}" for label, count in counts.items())


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=object).reshape(-1)
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one value per row: {vector.shape[0]} != {expected_length}.")
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
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    return int(parsed)
