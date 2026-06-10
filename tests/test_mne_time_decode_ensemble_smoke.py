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
                "source_calibration": kwargs["source_calibration"],
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
        if kwargs["alignment_method"] != "none":
            pd.DataFrame(
                [
                    {
                        "dataset": kwargs.get("dataset_name", ""),
                        "test_subject": decoder,
                        "alignment_method": kwargs["alignment_method"],
                        "alignment_anchor_mode": kwargs["alignment_anchor_mode"],
                        "alignment_anchor_column": kwargs["alignment_anchor_column"],
                        "sample_mode": kwargs["alignment_anchor_mode"],
                        "alignment_target_projection": kwargs["alignment_target_projection"],
                        "alignment_protocol": "strict_source_only",
                        "n_source_subjects": 2,
                        "n_source_rows": 20,
                        "source_anchor_value_source": "decoder_labels",
                        "n_source_anchor_values": 2,
                        "n_common_source_anchors": 2,
                        "source_anchor_rows_total": 20,
                        "source_anchor_rows_retained": 20,
                        "source_anchor_rows_dropped": 0,
                        "estimated_alignment_rows": 4,
                        "prefit_status": "ok",
                    }
                ]
            ).to_csv(kwargs["out_path"].parent / "alignment_anchor_availability.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "dataset": kwargs.get("dataset_name", ""),
                        "test_subject": decoder,
                        "alignment_method": kwargs["alignment_method"],
                        "sample_mode": kwargs["alignment_anchor_mode"],
                        "n_source_subjects": 2,
                        "n_classes": 2,
                        "n_alignment_rows": 4,
                        "requested_components": kwargs["alignment_components"],
                        "actual_components": 4,
                        "feature_dim": 8,
                        "decode_feature_dim": 4,
                        "uses_channel_projection_collapse": False,
                        "alignment_dimensionality_reduction": True,
                        "anchor_row_correlation_before": 0.1,
                        "anchor_row_correlation_after": 0.8,
                        "source_inner_decoding_before_alignment": 0.5,
                        "source_inner_decoding_after_alignment": 0.7,
                        "source_inner_raw_balanced_accuracy": 0.5,
                        "source_inner_aligned_balanced_accuracy": 0.7,
                        "source_inner_aligned_minus_raw": 0.2,
                        "source_inner_validation_type": "strict_source_loso_nearest_centroid_group_projection",
                        "target_transform_type": "source_group_projection",
                    }
                ]
            ).to_csv(kwargs["out_path"].parent / "alignment_diagnostics.csv", index=False)
        return frame

    monkeypatch.setattr("neureptrace.mne_time_decode_ensemble._run_time_resolved_decode", fake_source_decode)

    results = run_time_resolved_decode(
        epochs_path=tmp_path / "dummy-epo.fif",
        dataset_name="ensemble-demo",
        label_column="condition",
        out_path=tmp_path / "ensemble.csv",
        observation_out_path=tmp_path / "observations.csv",
        decoder="logistic-svm-ensemble",
        emission_mode="calibrated",
        decode_window=(0.12, 0.248),
        temporal_train_window=(0.12, 0.248),
        temporal_train_mode="pooled",
        class_prior_correction="train_uniform",
        source_calibration="temperature_plus_class_bias",
        source_time_selection="source_oof_best_time",
        source_time_selection_times=(0.088, 0.184, 0.280),
        source_time_selection_output_time=0.184,
        alignment_method="mcca",
        alignment_anchor_mode="class_repetition",
        alignment_anchor_column="stim_file",
        alignment_repetition_cap=12,
        alignment_components=32,
        alignment_times="same_decode_window",
        alignment_target_projection="group_projection",
        ensemble_source_decoders=("multinomial-logistic-weighted", "linear_svm", "shrinkage_lda"),
        ensemble_source_temperatures=(1.25, 1.0, 0.8),
        ensemble_score_mode="rank",
        ensemble_source_baseline_debiasing=True,
        ensemble_baseline_window=None,
    )

    assert len(calls) == 3
    assert {call["decoder"] for call in calls} == {"multinomial-logistic-weighted", "linear_svm", "shrinkage_lda"}
    assert all(call["decode_window"] == (0.12, 0.248) for call in calls)
    assert all(call["temporal_train_window"] == (0.12, 0.248) for call in calls)
    assert all(call["temporal_train_mode"] == "pooled" for call in calls)
    assert all(call["class_prior_correction"] == "train_uniform" for call in calls)
    assert all(call["source_calibration"] == "temperature_plus_class_bias" for call in calls)
    assert all(call["source_time_selection"] == "source_oof_best_time" for call in calls)
    assert all(call["source_time_selection_times"] == (0.088, 0.184, 0.280) for call in calls)
    assert all(call["source_time_selection_output_time"] == 0.184 for call in calls)
    assert all(call["alignment_method"] == "mcca" for call in calls)
    assert all(call["alignment_anchor_mode"] == "class_repetition" for call in calls)
    assert all(call["alignment_anchor_column"] == "stim_file" for call in calls)
    assert all(call["alignment_repetition_cap"] == 12 for call in calls)
    assert all(call["alignment_components"] == 32 for call in calls)
    assert all(call["alignment_times"] == "same_decode_window" for call in calls)
    assert all(call["alignment_target_projection"] == "group_projection" for call in calls)
    assert results["class_prior_correction"].unique().tolist() == ["train_uniform"]
    assert results["source_calibration"].unique().tolist() == ["temperature_plus_class_bias"]
    assert results["source_time_selection"].unique().tolist() == ["source_oof_best_time"]
    assert results["alignment_method"].unique().tolist() == ["mcca"]
    assert results["alignment_anchor_mode"].unique().tolist() == ["class_repetition"]
    assert results["alignment_anchor_column"].unique().tolist() == ["stim_file"]
    assert results["alignment_repetition_cap"].unique().tolist() == [12]
    assert results["alignment_components"].unique().tolist() == [32]
    assert results["alignment_times"].unique().tolist() == ["same_decode_window"]
    assert results["alignment_window_mode"].unique().tolist() == ["same_decode_window"]
    assert results["alignment_valid_for_benchmark"].unique().tolist() == [True]
    assert results["alignment_target_projection"].unique().tolist() == ["group_projection"]
    assert results["source_decoders"].unique().tolist() == ["multinomial-logistic-weighted|linear_svm|shrinkage_lda"]
    assert results["ensemble_weights"].unique().tolist() == ["0.333333333333|0.333333333333|0.333333333333"]
    assert results["ensemble_source_temperatures"].unique().tolist() == ["1.25|1|0.8"]
    assert results["ensemble_score_mode"].unique().tolist() == ["rank"]
    assert results["ensemble_source_baseline_debiasing"].unique().tolist() == [True]
    assert results["temporal_mode"].unique().tolist() == ["train_window_pooled"]
    diagnostics = pd.read_csv(tmp_path / "alignment_diagnostics.csv")
    assert diagnostics["dataset"].unique().tolist() == ["ensemble-demo"]
    assert set(diagnostics["test_subject"]) == {"multinomial-logistic-weighted", "linear_svm", "shrinkage_lda"}
    assert diagnostics["actual_components"].unique().tolist() == [4]
    assert diagnostics["source_inner_aligned_minus_raw"].unique().tolist() == [0.2]
    assert diagnostics["target_transform_type"].unique().tolist() == ["source_group_projection"]
    availability = pd.read_csv(tmp_path / "alignment_anchor_availability.csv")
    assert availability["dataset"].unique().tolist() == ["ensemble-demo"]
    assert set(availability["test_subject"]) == {"multinomial-logistic-weighted", "linear_svm", "shrinkage_lda"}
    assert availability["prefit_status"].unique().tolist() == ["ok"]
    source_observations = pd.read_csv(tmp_path / "ensemble_source_observations.csv")
    assert set(source_observations["decoder"]) == {"multinomial-logistic-weighted", "linear_svm", "shrinkage_lda"}


def test_logistic_svm_ensemble_marks_oracle_alignment_nonbenchmark(tmp_path, monkeypatch):
    def fake_source_decode(**kwargs):
        decoder = kwargs["decoder"]
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
                "probability_true_class": 0.8,
                "confidence": 0.8,
                "class_0": "class-0",
                "class_1": "class-1",
                "prob_class_0": 0.8 if label == 0 else 0.2,
                "prob_class_1": 0.2 if label == 0 else 0.8,
            }
            for index, label in enumerate((0, 1))
        ]
        kwargs["observation_out_path"].parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(kwargs["observation_out_path"], index=False)
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
                }
            ]
        )
        kwargs["out_path"].parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(kwargs["out_path"], index=False)
        return frame

    monkeypatch.setattr("neureptrace.mne_time_decode_ensemble._run_time_resolved_decode", fake_source_decode)

    results = run_time_resolved_decode(
        epochs_path=tmp_path / "dummy-epo.fif",
        label_column="condition",
        out_path=tmp_path / "ensemble.csv",
        decoder="logistic-svm-ensemble",
        emission_mode="calibrated",
        alignment_method="procrustes",
        alignment_target_projection="oracle_target_calibrated_alignment",
        ensemble_baseline_window=None,
    )

    assert results["alignment_target_projection"].unique().tolist() == ["oracle_target_calibrated_alignment"]
    assert results["alignment_oracle_target_calibrated"].unique().tolist() == [True]
    assert results["alignment_debug_upper_bound"].unique().tolist() == [True]
    assert results["alignment_valid_for_benchmark"].unique().tolist() == [False]
    assert results["alignment_protocol_note"].unique().tolist() == ["debug upper bound only; not valid for benchmark"]
