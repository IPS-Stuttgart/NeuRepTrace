"""Probabilistic tracing of neural representations over time."""

from __future__ import annotations

import importlib

__all__ = ["__version__"]
__version__ = "0.1.1"

from neureptrace import (  # noqa: E402
    _bushmeg_category2_autoencoder_config_patch,
    _bushmeg_source_loso_prototype_patch,
    _dataset_config_participant_patch,
    _decoding_adaptive_calibration,
    _decoding_c_grid_patch,
    _decoding_probability_patch,
    _decoding_regularization_patch,
    _event_detection_extensions,
    _metadata_column_validation_patch,
    _mne_alignment_calibration_anchor_patch,
    _observation_schema_label_patch,
    _observation_schema_probability_patch,
    _source_alignment_anchor_patch,
    _source_alignment_optimal_transport_patch,
    _source_alignment_oracle_patch,
    _source_alignment_pseudo_calibration_patch,
    _source_alignment_pseudo_repetition_patch,
)

_dataset_config_participant_patch.install()
_metadata_column_validation_patch.install()
_event_detection_extensions.install()
_decoding_regularization_patch.install()
_decoding_adaptive_calibration.install()
_decoding_c_grid_patch.install()
_decoding_probability_patch.install()
_observation_schema_probability_patch.install()
_observation_schema_label_patch.install()
_bushmeg_category2_autoencoder_config_patch.install()
_bushmeg_source_loso_prototype_patch.install()
_source_alignment_anchor_patch.install()
_source_alignment_pseudo_calibration_patch.install()
_source_alignment_pseudo_repetition_patch.install()
_source_alignment_optimal_transport_patch.install()
_mne_alignment_calibration_anchor_patch.install()

from neureptrace import (  # noqa: E402
    _source_alignment_contrastive_patch,
    _source_alignment_target_calibration_offsets_patch,
)

_source_alignment_contrastive_patch.install()
# Load source_alignment through the contrastive finder first, then apply the oracle
# wrapper to the loaded module. Otherwise the later oracle finder can mask the
# contrastive finder and leave method="contrastive" unregistered at runtime.
importlib.import_module("neureptrace.decoding.source_alignment")
_source_alignment_oracle_patch.install()
_source_alignment_target_calibration_offsets_patch.install()
