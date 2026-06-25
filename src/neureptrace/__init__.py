"""Probabilistic tracing of neural representations over time."""

from __future__ import annotations

import importlib

__all__ = ["__version__"]
__version__ = "0.1.1"

from . import (  # noqa: E402
    _alignment_window_config_patch,
    _bushmeg_all_protocols_prediction_metric_patch,
    _bushmeg_all_protocols_timeout_patch,
    _bushmeg_category2_autoencoder_config_patch,
    _bushmeg_source_loso_prototype_patch,
    _confusion_metadata_lookup_patch,
    _confusion_permutation_seed_patch,
    _correlation_prototype_sample_weight_patch,
    _dataset_config_participant_patch,
    _dataset_name_config_patch,
    _dataset_spec_csv_group_column_patch,
    _decoding_adaptive_calibration,
    _decoding_c_grid_patch,
    _decoding_classifier_param_patch,
    _decoding_grouped_cv_patch,
    _decoding_probability_patch,
    _decoding_regularization_patch,
    _event_detection_extensions,
    _few_shot_split_validation_patch,
    _few_shot_tuple_labels_patch,
    _mcca_repetition_count_patch,
    _metadata_column_validation_patch,
    _mne_alignment_calibration_anchor_patch,
    _mne_pseudo_alignment_fallback_validity_patch,
    _nll_eps_validation_patch,
    _observation_schema_label_patch,
    _observation_schema_probability_patch,
    _reconstruction_encoder_config_patch,
    _sample_weight_validation_patch,
    _response_window_time_validation_patch,
    _source_alignment_anchor_patch,
    _source_alignment_optimal_transport_patch,
    _source_alignment_oracle_patch,
    _source_alignment_pseudo_calibration_patch,
    _source_alignment_pseudo_repetition_patch,
    _source_alignment_times_validation_patch,
    _source_free_standardize_target_patch,
    _source_free_tuple_labels_patch,
    _source_mixstyle_tuple_labels_patch,
    _source_selection_class_balance_patch,
    _source_selection_temperature_patch,
    _tuple_label_calibration_split_patch,
)

_alignment_window_config_patch.install()
_dataset_config_participant_patch.install()
_dataset_name_config_patch.install()
_dataset_spec_csv_group_column_patch.install()
_metadata_column_validation_patch.install()
_confusion_metadata_lookup_patch.install()
_confusion_permutation_seed_patch.install()
_correlation_prototype_sample_weight_patch.install()
_event_detection_extensions.install()
_few_shot_split_validation_patch.install()
_few_shot_tuple_labels_patch.install()
_decoding_regularization_patch.install()
_decoding_adaptive_calibration.install()
_decoding_c_grid_patch.install()
_decoding_classifier_param_patch.install()
_decoding_grouped_cv_patch.install()
_decoding_probability_patch.install()
_observation_schema_probability_patch.install()
_observation_schema_label_patch.install()
_bushmeg_category2_autoencoder_config_patch.install()
_bushmeg_all_protocols_timeout_patch.install()
_bushmeg_all_protocols_prediction_metric_patch.install()
_reconstruction_encoder_config_patch.install()
_bushmeg_source_loso_prototype_patch.install()
_mcca_repetition_count_patch.install()
_source_alignment_anchor_patch.install()
_source_alignment_pseudo_calibration_patch.install()
_source_alignment_pseudo_repetition_patch.install()
_source_alignment_optimal_transport_patch.install()
_mne_alignment_calibration_anchor_patch.install()
_mne_pseudo_alignment_fallback_validity_patch.install()
_nll_eps_validation_patch.install()
_sample_weight_validation_patch.install()
_response_window_time_validation_patch.install()
_source_alignment_times_validation_patch.install()
_source_free_standardize_target_patch.install()
_source_free_tuple_labels_patch.install()
_source_mixstyle_tuple_labels_patch.install()
_source_selection_class_balance_patch.install()
_source_selection_temperature_patch.install()
_tuple_label_calibration_split_patch.install()

from . import (  # noqa: E402
    _source_alignment_cli_choices_patch,
    _source_alignment_contrastive_patch,
    _source_alignment_target_calibration_offsets_patch,
)

_source_alignment_contrastive_patch.install()
# Import source_alignment here so registered extension hooks compose in order.
# The contrastive extension must load before the later source-alignment wrapper.
# This keeps the method registration visible at runtime.
importlib.import_module("neureptrace.decoding.source_alignment")
_source_alignment_oracle_patch.install()
_source_alignment_target_calibration_offsets_patch.install()
_source_alignment_cli_choices_patch.install()
