"""Neutral in-memory dataset representation used by config-driven loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import pandas as pd


@dataclass(slots=True)
class EpochDataset:
    """Epoched M/EEG data plus metadata in a loader-independent form.

    The data array uses the same orientation as MNE ``Epochs`` objects:
    ``n_epochs × n_channels × n_times``. Loaders should normalize their native
    file formats into this representation before handing data to NeuRepTrace
    decoding workflows.
    """

    data: np.ndarray
    times: np.ndarray
    channel_names: list[str]
    metadata: pd.DataFrame
    name: str = "dataset"
    sensor_geometry: Any | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if _contains_boolean_values(self.data):
            raise ValueError("EpochDataset.data must contain numeric signal values, not boolean flags.")
        if _contains_boolean_values(self.times):
            raise ValueError("EpochDataset.times must contain numeric time values, not boolean flags.")
        if _contains_complex_values(self.data):
            raise ValueError("EpochDataset.data must contain real-valued signal values, not complex values.")
        if _contains_complex_values(self.times):
            raise ValueError("EpochDataset.times must contain real-valued time values, not complex values.")

        self.data = np.asarray(self.data, dtype=float)
        self.times = np.asarray(self.times, dtype=float)
        self.metadata = self.metadata.reset_index(drop=True).copy()
        self.channel_names = [str(channel_name) for channel_name in self.channel_names]

        if self.data.ndim != 3:
            raise ValueError("EpochDataset.data must have shape n_epochs × n_channels × n_times.")
        if self.times.ndim != 1:
            raise ValueError("EpochDataset.times must be one-dimensional.")
        if not np.all(np.isfinite(self.data)):
            raise ValueError("EpochDataset.data must contain only finite values.")
        if not np.all(np.isfinite(self.times)):
            raise ValueError("EpochDataset.times must contain only finite values.")
        duplicate_channel = _find_duplicate_string(self.channel_names)
        if duplicate_channel is not None:
            raise ValueError(f"EpochDataset.channel_names must be unique; duplicate channel name {duplicate_channel!r} found.")
        if self.data.shape[2] != len(self.times):
            raise ValueError(
                f"Data has {self.data.shape[2]} time points but times has {len(self.times)} entries."
            )
        if self.data.shape[1] != len(self.channel_names):
            raise ValueError(
                f"Data has {self.data.shape[1]} channels but {len(self.channel_names)} channel names were supplied."
            )
        if len(self.metadata) != self.data.shape[0]:
            raise ValueError(
                f"Metadata has {len(self.metadata)} rows but data has {self.data.shape[0]} epochs."
            )

    @property
    def X(self) -> np.ndarray:
        """Return the epoch data using feature-matrix terminology."""

        return self.data

    def with_channels(self, channel_names: list[str]) -> Self:
        """Return a dataset view containing channels in the requested order."""

        duplicate_requested_channel = _find_duplicate_string(channel_names)
        if duplicate_requested_channel is not None:
            raise ValueError(f"Requested channel names must be unique; duplicate channel name {duplicate_requested_channel!r} found.")
        index_by_name = {channel_name: index for index, channel_name in enumerate(self.channel_names)}
        missing = [channel_name for channel_name in channel_names if channel_name not in index_by_name]
        if missing:
            raise ValueError(f"Cannot select missing channels: {missing}.")
        indices = [index_by_name[channel_name] for channel_name in channel_names]
        provenance = dict(self.provenance)
        provenance["selected_channels"] = channel_names
        return type(self)(
            data=self.data[:, indices, :],
            times=self.times.copy(),
            channel_names=list(channel_names),
            metadata=self.metadata.copy(),
            name=self.name,
            sensor_geometry=self.sensor_geometry,
            provenance=provenance,
        )

    @staticmethod
    def _aligned_datasets(datasets: list[Self], channel_policy: str) -> tuple[list[Self], list[str], list[str]]:
        reference = datasets[0]
        if channel_policy == "exact":
            for dataset in datasets[1:]:
                if reference.channel_names != dataset.channel_names:
                    raise ValueError("Cannot concatenate datasets with different channel names or channel order.")
            return datasets, list(reference.channel_names), []

        if channel_policy == "first_dataset":
            target = list(reference.channel_names)
        elif channel_policy == "intersection":
            common = set(reference.channel_names)
            for dataset in datasets[1:]:
                common &= set(dataset.channel_names)
            target = [channel_name for channel_name in reference.channel_names if channel_name in common]
        else:
            raise ValueError("channel_policy must be 'exact', 'intersection', or 'first_dataset'.")

        if not target:
            raise ValueError("No common channels remain after applying channel alignment policy.")
        aligned = [dataset.with_channels(target) for dataset in datasets]
        dropped = sorted({channel for dataset in datasets for channel in dataset.channel_names if channel not in target})
        return aligned, target, dropped

    @classmethod
    def concatenate(cls, datasets: list[Self], *, name: str = "dataset", channel_policy: str = "exact") -> Self:
        """Concatenate compatible epoch datasets along the epoch axis.

        ``channel_policy='exact'`` is the safest default. ``intersection`` keeps
        the channel intersection in the first dataset's order. ``first_dataset``
        requires every later dataset to contain all first-dataset channels and
        drops any additional later-dataset channels.
        """

        if not datasets:
            raise ValueError("At least one dataset is required for concatenation.")

        reference = datasets[0]
        for dataset in datasets[1:]:
            if reference.times.shape != dataset.times.shape or not np.allclose(
                reference.times, dataset.times, rtol=0.0, atol=1e-9
            ):
                raise ValueError("Cannot concatenate datasets with different time axes.")

        aligned, channel_names, dropped_channels = cls._aligned_datasets(datasets, channel_policy)
        provenance = {
            "sources": [dataset.provenance for dataset in datasets],
            "dataset_names": [dataset.name for dataset in datasets],
            "channel_policy": channel_policy,
            "dropped_channels": dropped_channels,
        }
        return cls(
            data=np.concatenate([dataset.data for dataset in aligned], axis=0),
            times=reference.times.copy(),
            channel_names=channel_names,
            metadata=pd.concat([dataset.metadata for dataset in aligned], ignore_index=True),
            name=name,
            sensor_geometry=reference.sensor_geometry,
            provenance=provenance,
        )

    def infer_sampling_frequency(self) -> float:
        """Infer the sampling frequency from a uniformly sampled time vector."""

        if len(self.times) < 2:
            return 1.0
        diffs = np.diff(self.times)
        if np.any(diffs <= 0):
            raise ValueError("EpochDataset.times must be strictly increasing to infer sampling frequency.")
        median_step = float(np.median(diffs))
        if median_step <= 0:
            raise ValueError("Cannot infer sampling frequency from a non-positive time step.")
        if not np.allclose(diffs, median_step, rtol=1e-6, atol=1e-12):
            raise ValueError("EpochDataset.times must be uniformly sampled to infer sampling frequency.")
        return 1.0 / median_step

    def to_mne_epochs(self, *, channel_type: str | list[str] = "mag", sfreq: float | None = None):
        """Convert the neutral dataset into an in-memory MNE ``EpochsArray``."""

        import mne

        if sfreq is None:
            sfreq = self.infer_sampling_frequency()
        info = mne.create_info(self.channel_names, sfreq=float(sfreq), ch_types=channel_type)
        events = np.column_stack(
            [
                np.arange(len(self.metadata), dtype=int),
                np.zeros(len(self.metadata), dtype=int),
                np.ones(len(self.metadata), dtype=int),
            ]
        )
        return mne.EpochsArray(
            self.data,
            info,
            events=events,
            event_id={"event": 1},
            tmin=float(self.times[0]) if len(self.times) else 0.0,
            metadata=self.metadata,
            verbose="error",
        )


def _contains_boolean_values(values: object) -> bool:
    if isinstance(values, (bool, np.bool_)):
        return True
    try:
        array = np.asarray(values, dtype=object)
    except (TypeError, ValueError):
        return False
    if array.ndim == 0:
        return isinstance(array.item(), (bool, np.bool_))
    return any(isinstance(value, (bool, np.bool_)) for value in array.ravel(order="C"))


def _contains_complex_values(values: object) -> bool:
    if isinstance(values, (complex, np.complexfloating)):
        return True
    try:
        array = np.asarray(values, dtype=object)
    except (TypeError, ValueError):
        return False
    if array.ndim == 0:
        return isinstance(array.item(), (complex, np.complexfloating))
    return any(isinstance(value, (complex, np.complexfloating)) for value in array.ravel(order="C"))


def _find_duplicate_string(values: list[str]) -> str | None:
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text in seen:
            return text
        seen.add(text)
    return None
