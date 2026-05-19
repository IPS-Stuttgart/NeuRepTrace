from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from neureptrace.decoding.temporal_generalization import TemporalFeatureWindow


@dataclass(frozen=True)
class FeatureBlock:
    """Named feature-window collection for one subject and one logical data role.

    A block is the unit that downstream projects load from files and pass to
    NeuRepTrace workflows. Typical block names are dataset-specific (for example
    ``"main"`` or ``"cue"``); ``role`` gives workflows a dataset-independent way
    to find calibration/localizer data when such data are available.
    """

    name: str
    windows: Sequence[TemporalFeatureWindow]
    role: str = "analysis"
    metadata: pd.DataFrame | None = None
    sample_ids: Sequence[Any] | None = None
    groups: Sequence[Any] | None = None
    attributes: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("FeatureBlock.name must not be empty.")
        role = str(self.role).strip() or "analysis"
        windows = tuple(self.windows)
        if not windows:
            raise ValueError(f"FeatureBlock '{name}' must contain at least one window.")

        n_samples = _validate_block_windows(name, windows)
        if self.metadata is not None and len(self.metadata) != n_samples:
            raise ValueError(
                f"FeatureBlock '{name}' metadata rows must match samples: "
                f"{len(self.metadata)} != {n_samples}."
            )
        sample_ids = _normalize_optional_sequence(
            self.sample_ids,
            expected_length=n_samples,
            field_name="sample_ids",
            block_name=name,
        )
        groups = _normalize_optional_sequence(
            self.groups,
            expected_length=n_samples,
            field_name="groups",
            block_name=name,
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "attributes", dict(self.attributes or {}))

    @property
    def n_samples(self) -> int:
        """Number of samples/trials represented by every window in this block."""

        return len(self.windows[0].labels)

    @property
    def n_windows(self) -> int:
        """Number of time/feature windows in this block."""

        return len(self.windows)

    @property
    def labels(self) -> np.ndarray:
        """Labels in sample order for the block."""

        return np.asarray(self.windows[0].labels)

    @property
    def window_centers(self) -> tuple[float, ...]:
        """Window centers in seconds, preserving loader order."""

        return tuple(float(window.center) for window in self.windows)


@dataclass(frozen=True)
class SubjectFeatureSet:
    """All decoded feature blocks loaded for one subject or participant."""

    subject: str
    blocks: Mapping[str, FeatureBlock]
    default_block: str | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        subject = str(self.subject).strip()
        if not subject:
            raise ValueError("SubjectFeatureSet.subject must not be empty.")
        if not self.blocks:
            raise ValueError(f"SubjectFeatureSet '{subject}' must contain at least one block.")

        blocks: dict[str, FeatureBlock] = {}
        for key, block in self.blocks.items():
            if not isinstance(block, FeatureBlock):
                raise TypeError(f"SubjectFeatureSet '{subject}' block '{key}' must be a FeatureBlock.")
            block_name = str(key).strip()
            if block_name != block.name:
                raise ValueError(
                    f"SubjectFeatureSet '{subject}' block key '{block_name}' "
                    f"does not match block.name '{block.name}'."
                )
            blocks[block_name] = block

        default_block = self.default_block
        if default_block is None:
            default_block = next(iter(blocks))
        default_block = str(default_block).strip()
        if default_block not in blocks:
            available = ", ".join(sorted(blocks))
            raise KeyError(f"Unknown default block '{default_block}' for subject '{subject}'. Available blocks: {available}.")

        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "default_block", default_block)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def get_block(self, name: str | None = None) -> FeatureBlock:
        """Return a named block, or the subject's default block when omitted."""

        block_name = self.default_block if name is None else str(name).strip()
        assert block_name is not None
        try:
            return self.blocks[block_name]
        except KeyError as exc:
            available = ", ".join(sorted(self.blocks))
            raise KeyError(f"Unknown block '{block_name}' for subject '{self.subject}'. Available blocks: {available}.") from exc

    @property
    def calibration_blocks(self) -> tuple[FeatureBlock, ...]:
        """Blocks whose role is ``"calibration"``."""

        return tuple(block for block in self.blocks.values() if block.role == "calibration")


@dataclass(frozen=True)
class FeatureDataset:
    """Validated feature data returned by a NeuRepTrace-compatible loader."""

    subjects: Sequence[SubjectFeatureSet]
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        subjects = tuple(self.subjects)
        if not subjects:
            raise ValueError("FeatureDataset must contain at least one subject.")
        subject_ids = [subject.subject for subject in subjects]
        duplicates = sorted({subject for subject in subject_ids if subject_ids.count(subject) > 1})
        if duplicates:
            duplicate_text = ", ".join(duplicates)
            raise ValueError(f"FeatureDataset subject identifiers must be unique; duplicates: {duplicate_text}.")
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def subject_ids(self) -> tuple[str, ...]:
        """Subject identifiers in loader order."""

        return tuple(subject.subject for subject in self.subjects)

    def get_subject(self, subject: str) -> SubjectFeatureSet:
        """Return one subject by identifier."""

        subject_id = str(subject).strip()
        for candidate in self.subjects:
            if candidate.subject == subject_id:
                return candidate
        available = ", ".join(self.subject_ids)
        raise KeyError(f"Unknown subject '{subject_id}'. Available subjects: {available}.")

    def iter_blocks(self, *, name: str | None = None, role: str | None = None) -> Iterator[tuple[str, FeatureBlock]]:
        """Yield ``(subject_id, block)`` pairs matching an optional name or role."""

        normalized_role = None if role is None else str(role).strip()
        for subject in self.subjects:
            blocks = (subject.get_block(name),) if name is not None else tuple(subject.blocks.values())
            for block in blocks:
                if normalized_role is None or block.role == normalized_role:
                    yield subject.subject, block


class FeatureLoader(Protocol):
    """Protocol implemented by dataset-specific NeuRepTrace loader adapters."""

    def load(self) -> FeatureDataset:
        """Load and validate a dataset as generic feature blocks."""


FeatureLoaderCallable = Callable[[], FeatureDataset]


def load_feature_dataset(loader: FeatureDataset | FeatureLoader | FeatureLoaderCallable) -> FeatureDataset:
    """Resolve a dataset, loader object, or zero-argument loader function.

    Dataset-specific packages can implement ``FeatureLoader`` without depending
    on NeuRepTrace configuration parsing. Future YAML/JSON configs can select an
    adapter and still call this function to enforce a single validated contract.
    """

    if isinstance(loader, FeatureDataset):
        return loader
    if hasattr(loader, "load") and callable(loader.load):
        dataset = loader.load()
    elif callable(loader):
        dataset = loader()
    else:
        raise TypeError("loader must be a FeatureDataset, FeatureLoader, or zero-argument callable.")

    if not isinstance(dataset, FeatureDataset):
        raise TypeError(f"loader returned {type(dataset).__name__}, expected FeatureDataset.")
    return dataset


def _validate_block_windows(block_name: str, windows: tuple[TemporalFeatureWindow, ...]) -> int:
    first_labels = np.asarray(windows[0].labels)
    if first_labels.ndim != 1:
        raise ValueError(f"FeatureBlock '{block_name}' labels must be one-dimensional.")
    if len(first_labels) == 0:
        raise ValueError(f"FeatureBlock '{block_name}' must contain at least one sample.")

    n_samples = len(first_labels)
    for index, window in enumerate(windows):
        labels = np.asarray(window.labels)
        if labels.ndim != 1:
            raise ValueError(f"FeatureBlock '{block_name}' window {index} labels must be one-dimensional.")
        if len(labels) != n_samples:
            raise ValueError(
                f"FeatureBlock '{block_name}' window {index} label count must match the first window: "
                f"{len(labels)} != {n_samples}."
            )
        if not np.array_equal(labels, first_labels):
            raise ValueError(f"FeatureBlock '{block_name}' window {index} labels must match the first window order.")

        features = np.asarray(window.features)
        if features.ndim == 0:
            raise ValueError(f"FeatureBlock '{block_name}' window {index} features must have a sample axis.")
        if features.shape[0] != n_samples:
            raise ValueError(
                f"FeatureBlock '{block_name}' window {index} feature rows must match labels: "
                f"{features.shape[0]} != {n_samples}."
            )
    return n_samples


def _normalize_optional_sequence(
    values: Sequence[Any] | None,
    *,
    expected_length: int,
    field_name: str,
    block_name: str,
) -> tuple[Any, ...] | None:
    if values is None:
        return None
    normalized = tuple(values)
    if len(normalized) != expected_length:
        raise ValueError(
            f"FeatureBlock '{block_name}' {field_name} length must match samples: "
            f"{len(normalized)} != {expected_length}."
        )
    return normalized


__all__ = [
    "FeatureBlock",
    "FeatureDataset",
    "FeatureLoader",
    "FeatureLoaderCallable",
    "SubjectFeatureSet",
    "load_feature_dataset",
]
