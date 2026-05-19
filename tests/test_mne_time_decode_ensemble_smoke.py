import pytest

from neureptrace.mne_time_decode_ensemble import (
    ENSEMBLE_DECODER,
    ENSEMBLE_DECODER_CLI_CHOICES,
    normalize_time_decode_decoder_name,
    run_time_resolved_decode,
)


def test_logistic_svm_ensemble_aliases_are_exposed():
    assert "logistic-svm-ensemble" in ENSEMBLE_DECODER_CLI_CHOICES
    assert normalize_time_decode_decoder_name("calibrated-logistic-linear-svm-ensemble") == ENSEMBLE_DECODER


def test_logistic_svm_ensemble_requires_calibrated_emissions(tmp_path):
    with pytest.raises(ValueError, match="calibrated only"):
        run_time_resolved_decode(
            epochs_path=tmp_path / "dummy-epo.fif",
            label_column="condition",
            out_path=tmp_path / "out.csv",
            decoder="logistic-svm-ensemble",
            emission_mode="uncalibrated",
        )
