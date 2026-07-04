"""Strict source-only class prior helper."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_PRIOR_PROTOCOL = "strict_source_only_class_prior"
SOURCE_PRIOR_CATEGORY = "1_strict_source_only"
PRIOR_MODES = ("empirical", "uniform")


@dataclass(frozen=True, slots=True)
class SourcePriorResult:
    """Source-only class-prior estimate and provenance."""

    classes: np.ndarray
    prior: np.ndarray
    counts: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_source_class_prior(
    source_labels: Sequence[Any] | np.ndarray,
    *,
    classes: Sequence[Any] | np.ndarray | None = None,
    mode: str | None = "empirical",
) -> SourcePriorResult:
    """Estimate a class prior from source labels only."""

    labels = np.asarray(source_labels, dtype=object).reshape(-1)
    if labels.shape[0] < 1:
        raise ValueError("source_labels must contain at least one value.")
    class_values = _classes(labels, classes)
    resolved = normalize_prior_mode(mode)
    counts = np.asarray([np.count_nonzero(labels == class_label) for class_label in class_values.tolist()], dtype=int)
    if resolved == "uniform":
        prior = np.full(class_values.shape[0], 1.0 / class_values.shape[0], dtype=float)
    else:
        prior = counts.astype(float) / float(np.sum(counts))
    metadata = {
        "source_prior_protocol": SOURCE_PRIOR_PROTOCOL,
        "source_prior_protocol_category": SOURCE_PRIOR_CATEGORY,
        "source_prior_uses_source_labels": True,
        "source_prior_uses_heldout_features": False,
        "source_prior_uses_heldout_labels": False,
        "source_prior_valid_for_strict_source_only": True,
        "source_prior_valid_for_benchmark": True,
        "source_prior_n_rows": int(labels.shape[0]),
        "source_prior_n_classes": int(class_values.shape[0]),
        "source_prior_mode": resolved,
        "source_prior_class_counts": "|".join(f"{label}:{int(count)}" for label, count in zip(class_values.tolist(), counts, strict=True)),
    }
    return SourcePriorResult(classes=class_values, prior=prior.astype(np.float32, copy=False), counts=counts, metadata=metadata)


def normalize_prior_mode(value: str | None) -> str:
    """Normalize class-prior mode aliases."""

    normalized = "empirical" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"balanced": "uniform", "flat": "uniform", "frequency": "empirical", "counts": "empirical"}.get(normalized, normalized)
    if normalized not in PRIOR_MODES:
        raise ValueError(f"Unknown prior mode {value!r}. Available values: {', '.join(PRIOR_MODES)}.")
    return normalized


def _classes(labels: np.ndarray, classes: Sequence[Any] | np.ndarray | None) -> np.ndarray:
    if classes is None:
        values = tuple(dict.fromkeys(labels.tolist()))
    else:
        values = tuple(np.asarray(classes, dtype=object).reshape(-1).tolist())
    class_values = np.asarray(values, dtype=object)
    if class_values.shape[0] < 1:
        raise ValueError("classes must contain at least one value.")
    if len(set(class_values.tolist())) != class_values.shape[0]:
        raise ValueError("classes must be unique.")
    unknown = sorted({label for label in labels.tolist() if label not in set(class_values.tolist())}, key=repr)
    if unknown:
        raise ValueError(f"source_labels contain labels absent from classes: {unknown}.")
    return class_values
