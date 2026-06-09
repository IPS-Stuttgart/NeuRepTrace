"""Strict source-only feature alignment helpers.

The helpers in this module deliberately fit all supervised alignment anchors
from source subjects only. Held-out target rows are transformed with the
source-fitted group projection; target labels or cue labels are never accepted.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from neureptrace.decoding.hyperalignment_initialization import fit_class_hyperalignment
from neureptrace.decoding.mcca import fit_class_mcca

SOURCE_ALIGNMENT_METHODS = ("none", "procrustes", "hyperalignment", "mcca")
SOURCE_ALIGNMENT_ANCHOR_MODES = ("class_mean", "class_repetition")
SOURCE_ALIGNMENT_TARGET_PROJECTIONS = ("group_projection",)
DEFAULT_ALIGNMENT_TIMES = (0.088, 0.136, 0.184, 0.232, 0.280)
DEFAULT_ALIGNMENT_REPETITION_CAP = 16
DEFAULT_ALIGNMENT_COMPONENTS = 64


@dataclass(frozen=True, slots=True)
class SourceAlignmentConfig:
    """Configuration for strict source-only common-space alignment."""

    method: str = "none"
    anchor_mode: str = "class_mean"
    repetition_cap: int | None = DEFAULT_ALIGNMENT_REPETITION_CAP
    components: int | float = DEFAULT_ALIGNMENT_COMPONENTS
    times: tuple[float, ...] = DEFAULT_ALIGNMENT_TIMES
    target_projection: str = "group_projection"
    hyperalignment_iterations: int = 10
    mcca_regularization: float = 1e-6
    mcca_subject_pca_components: int | float | None = None

    @property
    def enabled(self) -> bool:
        return self.method != "none"

    def static_metadata(self) -> dict[str, Any]:
        return {
            "alignment_method": self.method,
            "alignment_anchor_mode": self.anchor_mode,
            "alignment_repetition_cap": "" if self.repetition_cap is None else int(self.repetition_cap),
            "alignment_components": self.components,
            "alignment_times": "|".join(f"{time:.6g}" for time in self.times),
            "alignment_target_projection": self.target_projection,
            "alignment_strict_source_only": bool(self.enabled),
        }


@dataclass(frozen=True, slots=True)
class SourceAlignmentResult:
    """Aligned train/test feature matrices and fold-local metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    metadata: dict[str, Any]


def normalize_source_alignment_method(method: str | None) -> str:
    normalized = "none" if method is None else str(method).strip().lower().replace("-", "_")
    if normalized in {"off", "false", "identity", "raw"}:
        normalized = "none"
    if normalized not in SOURCE_ALIGNMENT_METHODS:
        raise ValueError(
            f"Unknown source alignment method {method!r}. "
            f"Available methods: {', '.join(SOURCE_ALIGNMENT_METHODS)}."
        )
    return normalized


def normalize_source_alignment_anchor_mode(anchor_mode: str | None) -> str:
    normalized = "class_mean" if anchor_mode is None else str(anchor_mode).strip().lower().replace("-", "_")
    if normalized not in SOURCE_ALIGNMENT_ANCHOR_MODES:
        raise ValueError(
            f"Unknown source alignment anchor mode {anchor_mode!r}. "
            f"Available modes: {', '.join(SOURCE_ALIGNMENT_ANCHOR_MODES)}."
        )
    return normalized


def normalize_source_alignment_target_projection(target_projection: str | None) -> str:
    normalized = "group_projection" if target_projection is None else str(target_projection).strip().lower().replace("-", "_")
    if normalized not in SOURCE_ALIGNMENT_TARGET_PROJECTIONS:
        raise ValueError(
            "Strict source-only alignment only supports target_projection='group_projection'. "
            f"Got {target_projection!r}."
        )
    return normalized


def parse_alignment_times(times: Sequence[float] | str | None) -> tuple[float, ...]:
    if times is None:
        return DEFAULT_ALIGNMENT_TIMES
    if isinstance(times, str):
        text = times.strip()
        if not text:
            return DEFAULT_ALIGNMENT_TIMES
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        parts = [part.strip() for chunk in text.split(",") for part in chunk.split() if part.strip()]
        values = tuple(float(part) for part in parts)
    else:
        values = tuple(float(value) for value in times)
    if not values:
        raise ValueError("alignment_times must contain at least one time center.")
    if any(not np.isfinite(value) for value in values):
        raise ValueError("alignment_times must be finite.")
    return values


def normalize_alignment_repetition_cap(value: int | str | None) -> int | None:
    if value is None:
        return DEFAULT_ALIGNMENT_REPETITION_CAP
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "default"}:
            return DEFAULT_ALIGNMENT_REPETITION_CAP
        if text in {"none", "null", "all", "full"}:
            return None
        parsed = int(text)
    else:
        parsed = int(value)
    if parsed < 1:
        raise ValueError("alignment_repetition_cap must be positive, all, or none.")
    return parsed


def normalize_alignment_components(value: int | float | str | None) -> int | float:
    if value is None:
        return DEFAULT_ALIGNMENT_COMPONENTS
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "default"}:
            return DEFAULT_ALIGNMENT_COMPONENTS
        if text in {"inf", "infinity", "all", "full"}:
            return float("inf")
        parsed: int | float = float(text) if any(marker in text for marker in (".", "e")) else int(text)
    else:
        parsed = value
    if isinstance(parsed, float):
        if parsed == float("inf"):
            return parsed
        if not np.isfinite(parsed) or parsed <= 0:
            raise ValueError("alignment_components must be positive.")
        if parsed.is_integer():
            return int(parsed)
        raise ValueError("alignment_components must be an integer count or infinity.")
    parsed = int(parsed)
    if parsed < 1:
        raise ValueError("alignment_components must be positive.")
    return parsed


def source_alignment_config(
    *,
    method: str | None = None,
    anchor_mode: str | None = None,
    repetition_cap: int | str | None = DEFAULT_ALIGNMENT_REPETITION_CAP,
    components: int | float | str | None = DEFAULT_ALIGNMENT_COMPONENTS,
    times: Sequence[float] | str | None = None,
    target_projection: str | None = "group_projection",
    hyperalignment_iterations: int = 10,
    mcca_regularization: float = 1e-6,
    mcca_subject_pca_components: int | float | str | None = None,
) -> SourceAlignmentConfig:
    mcca_subject_components = None
    if mcca_subject_pca_components not in {None, "", "none", "None"}:
        mcca_subject_components = normalize_alignment_components(mcca_subject_pca_components)
    config = SourceAlignmentConfig(
        method=normalize_source_alignment_method(method),
        anchor_mode=normalize_source_alignment_anchor_mode(anchor_mode),
        repetition_cap=normalize_alignment_repetition_cap(repetition_cap),
        components=normalize_alignment_components(components),
        times=parse_alignment_times(times),
        target_projection=normalize_source_alignment_target_projection(target_projection),
        hyperalignment_iterations=int(hyperalignment_iterations),
        mcca_regularization=float(mcca_regularization),
        mcca_subject_pca_components=mcca_subject_components,
    )
    if config.hyperalignment_iterations < 1:
        raise ValueError("alignment hyperalignment_iterations must be positive.")
    if config.mcca_regularization < 0 or not np.isfinite(config.mcca_regularization):
        raise ValueError("alignment mcca_regularization must be finite and non-negative.")
    return config


def align_train_test_features(
    *,
    train_features: Sequence[Sequence[float]] | np.ndarray,
    train_labels: Sequence[Any] | np.ndarray,
    train_subject_ids: Sequence[Hashable] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceAlignmentConfig,
    target_labels: Sequence[Any] | np.ndarray | None = None,
) -> SourceAlignmentResult:
    """Fit source-only alignment and transform train/test feature rows.

    ``target_labels`` exists only as a guardrail: strict source-only alignment
    rejects it rather than silently accepting held-out labels.
    """

    if target_labels is not None:
        raise ValueError("Strict source-only alignment does not accept target labels.")

    train_matrix = _feature_matrix(train_features, name="train_features")
    test_matrix = _feature_matrix(test_features, name="test_features")
    train_vector = np.asarray(train_labels).reshape(-1)
    subject_vector = np.asarray(train_subject_ids, dtype=object).reshape(-1)
    if train_matrix.shape[0] != train_vector.shape[0]:
        raise ValueError("train_features and train_labels must have the same row count.")
    if train_matrix.shape[0] != subject_vector.shape[0]:
        raise ValueError("train_features and train_subject_ids must have the same row count.")
    if train_matrix.shape[1] != test_matrix.shape[1]:
        raise ValueError(
            "train_features and test_features must have the same feature width before alignment: "
            f"{train_matrix.shape[1]} != {test_matrix.shape[1]}."
        )

    metadata = config.static_metadata()
    if not config.enabled:
        return SourceAlignmentResult(
            train_features=train_matrix,
            test_features=test_matrix,
            metadata={
                **metadata,
                "alignment_n_components": "",
                "alignment_n_source_subjects": "",
                "alignment_n_classes": "",
                "alignment_repetitions_per_class": "",
            },
        )

    subject_ids = tuple(dict.fromkeys(subject_vector.tolist()))
    if len(subject_ids) < 2:
        raise ValueError("Strict source-only alignment requires at least two source subjects.")

    features_by_subject = {subject_id: train_matrix[subject_vector == subject_id] for subject_id in subject_ids}
    labels_by_subject = {subject_id: train_vector[subject_vector == subject_id] for subject_id in subject_ids}
    n_repetitions = _effective_repetitions_per_class(labels_by_subject, config)

    if config.method in {"procrustes", "hyperalignment"}:
        iterations = 1 if config.method == "procrustes" else config.hyperalignment_iterations
        model, alignment = fit_class_hyperalignment(
            features_by_subject,
            labels_by_subject,
            sample_mode=config.anchor_mode,
            n_repetitions_per_class=n_repetitions,
            n_components=config.components,
            n_iterations=iterations,
            initialization="mean" if config.method == "procrustes" else "pca",
        )
        transformed_by_subject = {subject_id: model.transform(subject_id, features_by_subject[subject_id]) for subject_id in subject_ids}
        transformed_test = model.transform_group(test_matrix)
        n_components = model.n_components
    elif config.method == "mcca":
        model, alignment = fit_class_mcca(
            features_by_subject,
            labels_by_subject,
            sample_mode=config.anchor_mode,
            n_repetitions_per_class=n_repetitions,
            n_components=config.components,
            regularization=config.mcca_regularization,
            subject_pca_components=config.mcca_subject_pca_components,
        )
        transformed_by_subject = {subject_id: model.transform(subject_id, features_by_subject[subject_id]) for subject_id in subject_ids}
        transformed_test = model.transform_group(test_matrix)
        n_components = model.n_components
    else:  # pragma: no cover - guarded by normalization
        raise ValueError(f"Unsupported source alignment method: {config.method}")

    transformed_train = np.empty((train_matrix.shape[0], transformed_test.shape[1]), dtype=float)
    for subject_id in subject_ids:
        transformed_train[subject_vector == subject_id] = transformed_by_subject[subject_id]

    return SourceAlignmentResult(
        train_features=transformed_train,
        test_features=transformed_test,
        metadata={
            **metadata,
            "alignment_n_components": int(n_components),
            "alignment_n_source_subjects": len(subject_ids),
            "alignment_n_classes": len(alignment.classes),
            "alignment_repetitions_per_class": "" if alignment.n_repetitions_per_class is None else int(alignment.n_repetitions_per_class),
        },
    )


def _effective_repetitions_per_class(
    labels_by_subject: Mapping[Hashable, np.ndarray],
    config: SourceAlignmentConfig,
) -> int | None:
    if config.anchor_mode != "class_repetition":
        return None
    subject_ids = tuple(labels_by_subject)
    first_classes = np.unique(labels_by_subject[subject_ids[0]])
    counts = []
    for subject_id in subject_ids:
        classes = np.unique(labels_by_subject[subject_id])
        if not np.array_equal(first_classes, classes):
            raise ValueError(f"Subject {subject_id!r} does not contain the common alignment classes.")
        counts.extend(int(np.sum(labels_by_subject[subject_id] == class_label)) for class_label in first_classes)
    available = min(counts)
    if available < 1:
        raise ValueError("Every source subject must have at least one sample per alignment class.")
    if config.repetition_cap is None:
        return int(available)
    return int(min(available, int(config.repetition_cap)))


def _feature_matrix(features: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values.")
    return matrix


__all__ = [
    "DEFAULT_ALIGNMENT_COMPONENTS",
    "DEFAULT_ALIGNMENT_REPETITION_CAP",
    "DEFAULT_ALIGNMENT_TIMES",
    "SOURCE_ALIGNMENT_ANCHOR_MODES",
    "SOURCE_ALIGNMENT_METHODS",
    "SourceAlignmentConfig",
    "SourceAlignmentResult",
    "align_train_test_features",
    "normalize_source_alignment_anchor_mode",
    "normalize_source_alignment_method",
    "source_alignment_config",
]
