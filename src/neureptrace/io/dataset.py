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
        self.data = np.asarray(self.data, dtype=float)
        self.times = np.asarray(self.times, dtype=float)
        self.metadata = self.metadata.reset_index(drop=True).copy()
        self.channel_names = [str(channel_name) for channel_name in self.channel_names]

        if self.data.ndim != 3:
            raise ValueError("EpochDataset.data must have shape n_epochs × n_channels × n_times.")
        if self.times.ndim != 1:
            raise ValueError("EpochDataset.times must be one-dimensional.")
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

    @classmethod
    def concatenate(cls, datasets: list[Self], *, name: str = "dataset") -> Self:
        """Concatenate compatible epoch datasets along the epoch axis."""

        if not datasets:
            raise ValueError("At least one dataset is required for concatenation.")

        reference = datasets[0]
        for dataset in datasets[1:]:
            if not np.allclose(reference.times, dataset.times, rtol=0.0, atol=1e-9):
                raise ValueError("Cannot concatenate datasets with different time axes.")
            if reference.channel_names != dataset.channel_names:
                raise ValueError("Cannot concatenate datasets with different channel names or channel order.")

        provenance = {
            "sources": [dataset.provenance for dataset in datasets],
            "dataset_names": [dataset.name for dataset in datasets],
        }
        return cls(
            data=np.concatenate([dataset.data for dataset in datasets], axis=0),
            times=reference.times.copy(),
            channel_names=list(reference.channel_names),
            metadata=pd.concat([dataset.metadata for dataset in datasets], ignore_index=True),
            name=name,
            sensor_geometry=reference.sensor_geometry,
            provenance=provenance,
        )

    def infer_sampling_frequency(self) -> float:
        """Infer the sampling frequency from the time vector."""

        if len(self.times) < 2:
            return 1.0
        diffs = np.diff(self.times)
        if np.any(diffs <= 0):
            raise ValueError("EpochDataset.times must be strictly increasing to infer sampling frequency.")
        median_step = float(np.median(diffs))
        if median_step <= 0:
            raise ValueError("Cannot infer sampling frequency from a non-positive time step.")
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
