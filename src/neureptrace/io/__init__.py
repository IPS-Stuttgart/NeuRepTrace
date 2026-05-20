"""Input adapters for external M/EEG dataset formats.

The modules in this package should contain dataset-independent conversion
logic only. Dataset- or paper-specific filename conventions belong in config
files or thin project repositories that call these adapters.
"""

from __future__ import annotations

from neureptrace.io.dataset import EpochDataset
from neureptrace.io.fieldtrip_mat import (
    FieldTripMatSpec,
    MetadataColumnSpec,
    ParticipantMatFiles,
    discover_participant_mat_files,
    load_fieldtrip_mat,
    load_fieldtrip_mat_epochs,
)

__all__ = [
    "EpochDataset",
    "FieldTripMatSpec",
    "MetadataColumnSpec",
    "ParticipantMatFiles",
    "discover_participant_mat_files",
    "load_fieldtrip_mat",
    "load_fieldtrip_mat_epochs",
]
