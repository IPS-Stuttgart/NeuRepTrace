"""Feature-matrix containers shared by dataset-specific loaders and decoders."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["FeatureDataset", "StackedFeatureSet", "SubjectFeatureSet"]


@dataclass(frozen=True)
class SubjectFeatureSet:
    """Feature matrix and trial annotations for one subject.

    Dataset-specific loaders should adapt their native file formats into this
    container before handing features to reusable NeuRepTrace decoding and
    reporting workflows. The container deliberately stores only generic
    feature-matrix information: rows are trials/observations, columns are
    decoder features, and optional arrays describe fold groups, trial indices,
    and per-trial metadata.
    """

    subject: Hashable
    features: Sequence[Sequence[float]] | np.ndarray
    labels: Sequence[Any] | np.ndarray
    groups: Sequence[Any] | np.ndarray | None = None
    trial_index: Sequence[Any] | np.ndarray | None = None
    metadata: pd.DataFrame | None = None
    feature_names: Sequence[Hashable] | None = None

    def __post_init__(self) -> None:
        _validate_hashable(self.subject, "subject")

        features = np.asarray(self.features)
        if features.ndim != 2:
            raise ValueError("features must be a 2D array with shape (n_trials, n_features).")
        if features.shape[0] == 0:
            raise ValueError("features must contain at least one trial.")
        if features.shape[1] == 0:
            raise ValueError("features must contain at least one feature column.")
        object.__setattr__(self, "features", features)

        labels = _as_1d_array(self.labels, "labels")
        _validate_hashable_array(labels, "labels")
        if labels.shape[0] != features.shape[0]:
            raise ValueError(f"labels has length {labels.shape[0]} but features has {features.shape[0]} rows.")
        object.__setattr__(self, "labels", labels)

        if self.groups is not None:
            groups = _as_1d_array(self.groups, "groups")
            _validate_hashable_array(groups, "groups")
            if groups.shape[0] != features.shape[0]:
                raise ValueError(f"groups has length {groups.shape[0]} but features has {features.shape[0]} rows.")
            object.__setattr__(self, "groups", groups)

        if self.trial_index is not None:
            trial_index = _as_1d_array(self.trial_index, "trial_index")
            _validate_hashable_array(trial_index, "trial_index")
            if trial_index.shape[0] != features.shape[0]:
                raise ValueError(f"trial_index has length {trial_index.shape[0]} but features has {features.shape[0]} rows.")
            object.__setattr__(self, "trial_index", trial_index)

        if self.metadata is not None:
            if not isinstance(self.metadata, pd.DataFrame):
                raise TypeError("metadata must be a pandas DataFrame when provided.")
            if len(self.metadata) != features.shape[0]:
                raise ValueError(f"metadata has {len(self.metadata)} rows but features has {features.shape[0]} rows.")
            object.__setattr__(self, "metadata", self.metadata.reset_index(drop=True).copy())

        if self.feature_names is not None:
            feature_names = tuple(self.feature_names)
            if len(feature_names) != features.shape[1]:
                raise ValueError(f"feature_names has length {len(feature_names)} but features has {features.shape[1]} columns.")
            for index, name in enumerate(feature_names):
                _validate_hashable(name, f"feature_names[{index}]")
            object.__setattr__(self, "feature_names", feature_names)

    @property
    def X(self) -> np.ndarray:
        """Alias for ``features`` used by scikit-learn-oriented callers."""

        return self.features

    @property
    def y(self) -> np.ndarray:
        """Alias for ``labels`` used by scikit-learn-oriented callers."""

        return self.labels

    @property
    def n_trials(self) -> int:
        """Number of rows in the feature matrix."""

        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        """Number of columns in the feature matrix."""

        return int(self.features.shape[1])

    def to_trial_frame(
        self,
        *,
        subject_column: str = "subject",
        label_column: str = "label",
        group_column: str = "group",
        trial_index_column: str = "trial_index",
        include_metadata: bool = True,
    ) -> pd.DataFrame:
        """Return one row per trial with generic annotations and optional metadata."""

        frame = pd.DataFrame(
            {
                subject_column: [self.subject] * self.n_trials,
                label_column: self.labels,
            }
        )
        if self.groups is not None:
            frame[group_column] = self.groups
        if self.trial_index is not None:
            frame[trial_index_column] = self.trial_index
        if include_metadata and self.metadata is not None:
            _raise_if_column_overlap(frame.columns, self.metadata.columns)
            frame = pd.concat([frame, self.metadata.reset_index(drop=True)], axis=1)
        return frame


@dataclass(frozen=True)
class StackedFeatureSet:
    """Concatenated feature arrays from one or more subjects."""

    features: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    groups: np.ndarray | None = None
    trial_index: np.ndarray | None = None

    @property
    def X(self) -> np.ndarray:
        """Alias for ``features`` used by scikit-learn-oriented callers."""

        return self.features

    @property
    def y(self) -> np.ndarray:
        """Alias for ``labels`` used by scikit-learn-oriented callers."""

        return self.labels


@dataclass(frozen=True)
class FeatureDataset:
    """Collection of per-subject feature matrices."""

    subjects: Sequence[SubjectFeatureSet]
    name: str | None = None

    def __post_init__(self) -> None:
        subject_sets = tuple(self.subjects)
        if not subject_sets:
            raise ValueError("FeatureDataset requires at least one SubjectFeatureSet.")
        if not all(isinstance(subject_set, SubjectFeatureSet) for subject_set in subject_sets):
            raise TypeError("subjects must contain only SubjectFeatureSet instances.")

        seen_subjects: set[Hashable] = set()
        duplicates: list[Hashable] = []
        for subject_set in subject_sets:
            if subject_set.subject in seen_subjects:
                duplicates.append(subject_set.subject)
            seen_subjects.add(subject_set.subject)
        if duplicates:
            duplicate_text = ", ".join(repr(subject) for subject in duplicates)
            raise ValueError(f"FeatureDataset subject identifiers must be unique; duplicates: {duplicate_text}.")

        object.__setattr__(self, "subjects", subject_sets)

    @property
    def subject_ids(self) -> tuple[Hashable, ...]:
        """Subject identifiers in dataset order."""

        return tuple(subject_set.subject for subject_set in self.subjects)

    @property
    def n_subjects(self) -> int:
        """Number of subject-level feature sets."""

        return len(self.subjects)

    @property
    def n_trials(self) -> int:
        """Total number of trials across all subjects."""

        return sum(subject_set.n_trials for subject_set in self.subjects)

    @property
    def feature_dimensions(self) -> dict[Hashable, int]:
        """Per-subject feature counts."""

        return {subject_set.subject: subject_set.n_features for subject_set in self.subjects}

    @property
    def has_uniform_feature_count(self) -> bool:
        """Whether all subject feature matrices have the same number of columns."""

        return len(set(self.feature_dimensions.values())) == 1

    @property
    def n_features(self) -> int:
        """Common feature count, raising when subjects have incompatible feature dimensions."""

        dimensions = set(self.feature_dimensions.values())
        if len(dimensions) != 1:
            raise ValueError(f"FeatureDataset has non-uniform feature counts: {self.feature_dimensions}.")
        return int(next(iter(dimensions)))

    def get_subject(self, subject: Hashable) -> SubjectFeatureSet:
        """Return the feature set for one subject identifier."""

        for subject_set in self.subjects:
            if subject_set.subject == subject:
                return subject_set
        raise KeyError(subject)

    def select_subjects(self, subjects: Iterable[Hashable]) -> "FeatureDataset":
        """Return a dataset restricted to selected subject identifiers in the requested order."""

        selected = tuple(self.get_subject(subject) for subject in subjects)
        return FeatureDataset(selected, name=self.name)

    def drop_subjects(self, subjects: Iterable[Hashable]) -> "FeatureDataset":
        """Return a dataset excluding selected subject identifiers."""

        excluded = set(subjects)
        selected = tuple(subject_set for subject_set in self.subjects if subject_set.subject not in excluded)
        return FeatureDataset(selected, name=self.name)

    def stack(self, subjects: Iterable[Hashable] | None = None) -> StackedFeatureSet:
        """Concatenate selected subjects into arrays suitable for fold-level decoders.

        All selected subjects must have the same number of feature columns. The
        returned ``subjects`` array records the source subject for each row.
        ``groups`` and ``trial_index`` are included only when all selected
        subjects provide them.
        """

        selected = self.subjects if subjects is None else self.select_subjects(subjects).subjects
        if not selected:
            raise ValueError("At least one subject is required for stacking.")
        _validate_uniform_feature_count(selected)

        features = np.vstack([subject_set.features for subject_set in selected])
        labels = np.concatenate([subject_set.labels for subject_set in selected])
        subject_ids = np.concatenate([_repeat_object(subject_set.subject, subject_set.n_trials) for subject_set in selected])
        groups = np.concatenate([subject_set.groups for subject_set in selected]) if all(subject_set.groups is not None for subject_set in selected) else None
        trial_index = (
            np.concatenate([subject_set.trial_index for subject_set in selected]) if all(subject_set.trial_index is not None for subject_set in selected) else None
        )
        return StackedFeatureSet(features=features, labels=labels, subjects=subject_ids, groups=groups, trial_index=trial_index)

    def to_trial_frame(self, *, include_metadata: bool = True) -> pd.DataFrame:
        """Return a concatenated per-trial annotation table."""

        return pd.concat([subject_set.to_trial_frame(include_metadata=include_metadata) for subject_set in self.subjects], ignore_index=True)


def _as_1d_array(values: Sequence[Any] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 and not isinstance(values, np.ndarray):
        array = _object_array(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    return array


def _object_array(values: Sequence[Any]) -> np.ndarray:
    sequence = list(values)
    array = np.empty(len(sequence), dtype=object)
    for index, value in enumerate(sequence):
        array[index] = value
    return array


def _repeat_object(value: Any, count: int) -> np.ndarray:
    array = np.empty(count, dtype=object)
    for index in range(count):
        array[index] = value
    return array


def _validate_hashable(value: Any, name: str) -> None:
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be hashable.") from exc


def _validate_hashable_array(values: np.ndarray, name: str) -> None:
    for index, value in enumerate(values.tolist()):
        _validate_hashable(value, f"{name}[{index}]")


def _raise_if_column_overlap(left: Iterable[Hashable], right: Iterable[Hashable]) -> None:
    overlap = set(left).intersection(right)
    if overlap:
        overlap_text = ", ".join(repr(column) for column in sorted(overlap, key=repr))
        raise ValueError(f"metadata columns overlap generated trial-frame columns: {overlap_text}.")


def _validate_uniform_feature_count(subjects: Sequence[SubjectFeatureSet]) -> None:
    dimensions = {subject_set.subject: subject_set.n_features for subject_set in subjects}
    if len(set(dimensions.values())) != 1:
        raise ValueError(f"Subjects must have the same number of feature columns to stack; got {dimensions}.")
