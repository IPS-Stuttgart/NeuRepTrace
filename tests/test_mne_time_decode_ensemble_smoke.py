import pandas as pd
import pytest

from neureptrace.mne_time_decode_ensemble import (
    ENSEMBLE_DECODER,
    ENSEMBLE_DECODER_CLI_CHOICES,
    _SOURCE_DECODERS,
    _parse_source_temperatures,
    _parse_weights,
    _parse_source_decoders,
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


def test_logistic_svm_ensemble_accepts_weighted_source_decoder_override():
    requested, normalized = _parse_source_decoders(("multinomial-logistic-weighted", "linear-svm", "shrinkage-lda"))

    assert requested == ("multinomial-logistic-weighted", "linear-svm", "shrinkage-lda")
    assert normalized == ("multinomial-logistic-weighted", "linear_svm", "shrinkage_lda")


def test_logistic_svm_ensemble_uses_equal_weights_for_nondefault_sources():
    assert _parse_weights(None, 3) == (1.0, 1.0, 1.0)
    assert _parse_weights((0.5, 0.3, 0.2), 3) == (0.5, 0.3, 0.2)


def test_logistic_svm_ensemble_parses_source_temperatures():
    assert _parse_source_temperatures(None, 3) == (1.0, 1.0, 1.0)
    assert _parse_source_temperatures((1.25, 1.0, 0.8), 3) == (1.25, 1.0, 0.8)


def test_logistic_svm_ensemble_requires_calibrated_emissions(tmp_path):
    with pytest.raises(ValueError, match="calibrated only"):
        run_time_resolved_decode(
            epochs_path=tmp_path / "dummy-epo.fif",
            label_column="condition",
            out_path=tmp_path / "out.csv",
            decoder="logistic-svm-ensemble",
            emission_mode="uncalibrated",
        )


def test_logistic_svm_ensemble_passes_window_controls_to_source_decoders(tmp_path, monkeypatch):
    calls = []

    def fake_source_decode(**kwargs):
        calls.append(kwargs)
        decoder = kwargs["decoder"]
        probabilities_by_decoder = {
            "multinomial-logistic-weighted": 0.8,
            "linear_svm": 0.7,
            "shrinkage_lda": 0.6,
        }
        probability = probabilities_by_decoder[decoder]
        rows = [
            {
                "subject": "sub-01",
                "fold": 0,
                "decoder": decoder,
                "emission_mode": "calibrated",
                "time": 0.184,
                "window_start": 0.134,
                "window_stop": 0.234,
                "sample_index": index,
                "sequence_id": index,
                "true_label": label,
                "true_class": f"class-{label}",
                "predicted_label": label,
                "predicted_class": f"class-{label}",
                "probability_true_class": probability,
                "confidence": probability,
                "class_0": "class-0",
                "class_1": "class-1",
                "prob_class_0": probability if label == 0 else 1.0 - probability,
                "prob_class_1": 1.0 - probability if label == 0 else probability,
                "class_prior_correction": kwargs["class_prior_correction"],
            }
            for index, label in enumerate((0, 1))
        ]
        kwargs["observation_out_path"].parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(kwargs["observation_out_path"], index=False)
        kwargs["out_path"].parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            [
                {
                    "fold": 0,
                    "decoder": decoder,
                    "emission_mode": "calibrated",
                    "time": 0.184,
                    "window_start": 0.134,
                    "window_stop": 0.234,
                    "accuracy": 1.0,
                    "balanced_accuracy": 1.0,
                    "top2_accuracy": 1.0,
                    "top3_accuracy": 1.0,
                    "log_loss": 0.2,
                    "brier": 0.1,
                    "ece": 0.0,
                    "n_test": 2,
                    "temporal_mode": "train_window_pooled",
                    "temporal_train_window_start": 0.12,
                    "temporal_train_window_stop": 0.248,
                }
            ]
        )
        frame.to_csv(kwargs["out_path"], index=False)
        return frame

    monkeypatch.setattr("neureptrace.mne_time_decode_ensemble._run_time_resolved_decode", fake_source_decode)

    results = run_time_resolved_decode(
        epochs_path=tmp_path / "dummy-epo.fif",
        label_column="condition",
        out_path=tmp_path / "ensemble.csv",
        decoder="logistic-svm-ensemble",
        emission_mode="calibrated",
        decode_window=(0.12, 0.248),
        temporal_train_window=(0.12, 0.248),
        temporal_train_mode="pooled",
        class_prior_correction="train_uniform",
        ensemble_source_decoders=("multinomial-logistic-weighted", "linear_svm", "shrinkage_lda"),
        ensemble_source_temperatures=(1.25, 1.0, 0.8),
        ensemble_baseline_window=None,
    )

    assert len(calls) == 3
    assert {call["decoder"] for call in calls} == {"multinomial-logistic-weighted", "linear_svm", "shrinkage_lda"}
    assert all(call["decode_window"] == (0.12, 0.248) for call in calls)
    assert all(call["temporal_train_window"] == (0.12, 0.248) for call in calls)
    assert all(call["temporal_train_mode"] == "pooled" for call in calls)
    assert all(call["class_prior_correction"] == "train_uniform" for call in calls)
    assert results["class_prior_correction"].unique().tolist() == ["train_uniform"]
    assert results["source_decoders"].unique().tolist() == ["multinomial-logistic-weighted|linear_svm|shrinkage_lda"]
    assert results["ensemble_weights"].unique().tolist() == ["0.333333333333|0.333333333333|0.333333333333"]
    assert results["ensemble_source_temperatures"].unique().tolist() == ["1.25|1|0.8"]
    assert results["temporal_mode"].unique().tolist() == ["train_window_pooled"]
