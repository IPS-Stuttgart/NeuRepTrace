"""Probabilistic tracing of neural representations over time."""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.1"

from neureptrace import (  # noqa: E402
    _dataset_config_participant_patch,
    _decoding_adaptive_calibration,
    _decoding_c_grid_patch,
    _decoding_probability_patch,
    _decoding_regularization_patch,
    _event_detection_extensions,
    _metadata_column_validation_patch,
    _observation_schema_probability_patch,
)

_dataset_config_participant_patch.install()
_metadata_column_validation_patch.install()
_event_detection_extensions.install()
_decoding_regularization_patch.install()
_decoding_adaptive_calibration.install()
_decoding_c_grid_patch.install()
_decoding_probability_patch.install()
_observation_schema_probability_patch.install()
