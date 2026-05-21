import pandas as pd
import pytest

from neureptrace.mne_time_decode_ensemble import (
    ENSEMBLE_DECODER,
    ENSEMBLE_DECODER_CLI_CHOICES,
    _SOURCE_DECODERS,
    normalize_time_decode_decoder_name,
    run_time_resolved_decode,
)
from neureptrace.observation_ensemble import ensemble_probability_observations


def test_logistic_svm_ensemble_aliases_are_exposed():
    assert "logistic-svm-ensemble" in ENSEMBLE_DECODER_CLI_CHOICES
    assert normalize_time_decode_decoder_name("calibrated-logistic-linear-svm-ensemble") == ENSEMBLE_DECODER


def test_logistic_svm_ensemble_source_names_match_normalized_observations():
    observations = pd.DataFrame(
        [
            {
                "decoder": decoder,
                "emission_mode": "calibrated",
                "time": -0.1,
                "true_label": 0,
                "fold": 0,
                "sample_index": 0,
                "prob_class_0": prob_0,
                "prob_class_1": 1.0 - prob_0,
            }
            for decoder, prob_0 in zip(_SOURCE_DECODERS, (0.8, 0.6), strict=True)
        ]
    )

    ensemble = ensemble_probability_observations(
        observations,
        decoders=_SOURCE_DECODERS,
        source_emission_mode="calibrated",
        baseline_window=None,
        output_decoder=ENSEMBLE_DECODER,
    )

    assert ensemble["source_decoders"].iloc[0] == "|".join(_SOURCE_DECODERS)
    assert ensemble["decoder"].iloc[0] == ENSEMBLE_DECODER


def test_logistic_svm_ensemble_requires_calibrated_emissions(tmp_path):
    with pytest.raises(ValueError, match="calibrated only"):
        run_time_resolved_decode(
            epochs_path=tmp_path / "dummy-epo.fif",
            label_column="condition",
            out_path=tmp_path / "out.csv",
            decoder="logistic-svm-ensemble",
            emission_mode="uncalibrated",
        )
