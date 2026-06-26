"""Source-class balancing for Joint Distribution Adaptation.

The balancing step uses source rows and source labels only. The wrapped JDA fit
then uses unlabeled target features, so the complete pipeline is Category 2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from neureptrace.decoding.joint_distribution_adaptation import (
    JointDistributionAdaptationResult,
    fit_joint_distribution_adaptation,
)

CLASS_BALANCED_JDA_PROTOCOL = "unlabeled_target_class_balanced_jda"
CLASS_BALANCED_JDA_CATEGORY = "2_unlabeled_target_adaptive"
CLASS_BALANCE_STRATEGIES = ("oversample", "undersample", "median")


@dataclass(frozen=True, slots=True)
class ClassBalanceResampleResult:
    """Fold-local source-row resampling result."""

    indices: np.ndarray
    classes: tuple[Any, ...]
    original_counts: tuple[int, ...]
    balanced_counts: tuple[int, ...]
    strategy: str
    random_state: int | None


def class_balanced_source_indices(
    source_labels: Sequence[Any] | np.ndarray,
    *,
    strategy: str = "oversample",
    random_state: int | None = 13,
) -> ClassBalanceResampleResult:
    """Return deterministic class-balanced source-row indices."""

    labels = _object_vector(source_labels, name="source_labels")
    classes = tuple(dict.fromkeys(labels.tolist()))
    if len(classes) < 2:
        raise ValueError("class balancing requires at least two source classes")
    mode = normalize_class_balance_strategy(strategy)
    class_rows = [_matching_indices(labels, class_label) for class_label in classes]
    counts = np.asarray([rows.size for rows in class_rows], dtype=int)
    target_count = _target_count(counts, mode)
    rng = np.random.default_rng(random_state)
    sampled: list[np.ndarray] = []
    for rows in class_rows:
        if rows.size == target_count:
            selected = rows.copy()
        elif rows.size < target_count:
            selected = rng.choice(rows, size=target_count, replace=True)
        else:
            selected = rng.choice(rows, size=target_count, replace=False)
        sampled.append(np.asarray(selected, dtype=int))
    indices = np.concatenate(sampled)
    rng.shuffle(indices)
    balanced_counts = tuple(int(np.count_nonzero(_object_mask(labels[indices], value))) for value in classes)
    return ClassBalanceResampleResult(
        indices=indices,
        classes=classes,
        original_counts=tuple(int(value) for value in counts),
        balanced_counts=balanced_counts,
        strategy=mode,
        random_state=random_state,
    )


def fit_class_balanced_jda(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    balance_strategy: str = "oversample",
    balance_random_state: int | None = 13,
    target_probabilities: Sequence[Sequence[float]] | np.ndarray | None = None,
    **jda_kwargs: Any,
) -> JointDistributionAdaptationResult:
    """Fit JDA after fold-local source-class balancing.

    The public API intentionally has no target-label argument. Source labels are
    used for resampling and JDA supervision; target rows remain unlabeled.
    """

    features = _feature_matrix(source_features, name="source_features")
    labels = _object_vector(source_labels, name="source_labels")
    if labels.shape[0] != features.shape[0]:
        raise ValueError("source_labels must contain one value per source row")
    resample = class_balanced_source_indices(
        labels,
        strategy=balance_strategy,
        random_state=balance_random_state,
    )
    result = fit_joint_distribution_adaptation(
        features[resample.indices],
        labels[resample.indices],
        target_features,
        target_probabilities=target_probabilities,
        **jda_kwargs,
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "class_balanced_jda": True,
            "class_balanced_jda_protocol": CLASS_BALANCED_JDA_PROTOCOL,
            "class_balanced_jda_protocol_category": CLASS_BALANCED_JDA_CATEGORY,
            "class_balanced_jda_strategy": resample.strategy,
            "class_balanced_jda_random_state": "" if resample.random_state is None else int(resample.random_state),
            "class_balanced_jda_original_source_rows": int(features.shape[0]),
            "class_balanced_jda_resampled_source_rows": int(resample.indices.shape[0]),
            "class_balanced_jda_original_counts": "|".join(str(value) for value in resample.original_counts),
            "class_balanced_jda_balanced_counts": "|".join(str(value) for value in resample.balanced_counts),
            "class_balanced_jda_uses_source_labels": True,
            "class_balanced_jda_uses_target_features": True,
            "class_balanced_jda_uses_target_labels": False,
            "class_balanced_jda_valid_for_strict_source_only": False,
            "class_balanced_jda_valid_for_unlabeled_target_adaptation": True,
        }
    )
    return replace(result, metadata=metadata)


def normalize_class_balance_strategy(value: str | None) -> str:
    """Normalize class-balancing strategy aliases."""

    normalized = "oversample" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {
        "over": "oversample",
        "upsample": "oversample",
        "max": "oversample",
        "under": "undersample",
        "downsample": "undersample",
        "min": "undersample",
        "median_count": "median",
    }.get(normalized, normalized)
    if normalized not in CLASS_BALANCE_STRATEGIES:
        raise ValueError(
            f"Unknown class-balance strategy {value!r}. "
            f"Available strategies: {', '.join(CLASS_BALANCE_STRATEGIES)}."
        )
    return normalized


def _target_count(counts: np.ndarray, strategy: str) -> int:
    if strategy == "oversample":
        return int(np.max(counts))
    if strategy == "undersample":
        return int(np.min(counts))
    return max(1, int(np.rint(np.median(counts))))


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values")
    return matrix


def _object_vector(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence") from exc
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = _hashable_label(value, name=name)
    return vector


def _hashable_label(value: Any, *, name: str) -> Any:
    try:
        hash(value)
    except TypeError:
        return _hashable_composite_label(value, name=name)
    return value


def _hashable_composite_label(value: Any, *, name: str) -> tuple[Any, ...]:
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} values must be hashable") from exc
    label = tuple(_hashable_label(item, name=name) for item in items)
    try:
        hash(label)
    except TypeError as exc:  # Defensive guard for unusual nested containers.
        raise ValueError(f"{name} values must be hashable") from exc
    return label


def _object_mask(values: np.ndarray, target: Any) -> np.ndarray:
    return np.asarray([_object_equal(value, target) for value in values.tolist()], dtype=bool)


def _matching_indices(values: np.ndarray, target: Any) -> np.ndarray:
    return np.flatnonzero(_object_mask(values, target))


def _object_equal(left: Any, right: Any) -> bool:
    result = left == right
    if isinstance(result, np.ndarray):
        return bool(np.array_equal(left, right))
    return bool(result)
