"""Input adapters for external M/EEG dataset formats."""

from __future__ import annotations

from neureptrace.io import _fieldtrip_bool_config_patch, _fieldtrip_config_participant

_fieldtrip_config_participant.install()
_fieldtrip_bool_config_patch.install()

from neureptrace.io.dataset import EpochDataset  # noqa: E402
from neureptrace.io.fieldtrip_mat import (  # noqa: E402
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
