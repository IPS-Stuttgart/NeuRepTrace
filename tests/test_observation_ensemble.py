from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.observation_ensemble import (
    DEFAULT_ENSEMBLE_DECODER,
    ensemble_probability_observations,
    main,
    summarize_ensemble_metrics,
)


def _source_observations() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    decoder_specs = {
        "logistic": {
            "baseline": (0.70, 0.30),
            "effect": (0.80, 0.20),
        },
        "linear_svm": {
            "baseline": (0.80, 0.20),
            "effect": (0.95, 0.05),
        },
    }
    for decoder, probabilities_by_window in decoder_specs.items():
        for sample_index, true_label in enumerate([0, 1]):
            rows.append(
                {
                    "subject": "sub-01",
                    "fold": 0,
                    "split_id": "split-0",
                    "seed": 13,
                    "decoder": decoder,
                    "backend": "sklearn",
                    "emission_mode": "calibrated",
                    "train_time": -0.20,
                    "test_time": -0.20,
                    "time": -0.20,
                    "window_start": -0.21,
                    "window_stop": -0.19,
                    "sample_index": sample_index,
                    "sequence_id": sample_index,
                    "true_label": true_label,
                    "true_class": "zero" if true_label == 0 else "one",
                    "class_0": "zero",
                    "class_1": "one",
                    "prob_class_0": probabilities_by_window["baseline"][0],
                    "prob_class_1": probabilities_by_window["baseline"][1],
                }
            )
        for sample_index in [0, 1]:
            rows.append(
                {
                    "subject": "sub-01",
                    "fold": 0,
                    "split_id": "split-0",
                    "seed": 13,
                    "decoder": decoder,
                    "backend": "sklearn",
                    "emission_mode": "calibrated",
                    "train_time": 0.10,
                    "test_time": 0.10,
                    "time": 0.10,
                    "window_start": 0.09,
                    "window_stop": 0.11,
                    "sample_index": sample_index,
                    "sequence_id": sample_index,
                    "true_label": 0,
                    "true_class": "zero",
                    "class_0": "zero",
                    "class_1": "one",
                    "prob_class_0": probabilities_by_window["effect"][0],
                    "prob_class_1": probabilities_by_window["effect"][1],
                }
            )
    return pd.DataFrame(rows)


def test_ensemble_probability_observations_baseline_debiases_bias() -> None:
    ensemble = ensemble_probability_observations(
        _source_observations(),
        baseline_window=(-0.25, -0.15),
    )

    assert ensemble["decoder"].unique().tolist() == [DEFAULT_ENSEMBLE_DECODER]
    assert ensemble["backend"].unique().tolist() == ["ensemble"]
    assert ensemble["source_decoders"].unique().tolist() == ["logistic|linear_svm"]
    assert ensemble["n_baseline_observations"].unique().tolist() == [2]
    assert np.allclose(ensemble[["prob_class_0", "prob_class_1"]].sum(axis=1), 1.0)

    baseline = ensemble.loc[ensemble["time"] == -0.20]
    effect = ensemble.loc[ensemble["time"] == 0.10]
    assert np.allclose(baseline["prob_class_0"], 0.5)
    assert effect["prob_class_0"].gt(0.70).all()
    assert effect["predicted_label"].tolist() == [0, 0]
    assert effect["probability_true_class"].gt(0.70).all()


def test_ensemble_probability_observations_accepts_hyphenated_decoder_aliases() -> None:
    observations = _source_observations().replace({"decoder": {"logistic": "multinomial_logistic"}})

    ensemble = ensemble_probability_observations(
        observations,
        decoders=("multinomial-logistic", "linear_svm"),
        baseline_window=(-0.25, -0.15),
    )

    assert ensemble["source_decoders"].unique().tolist() == ["multinomial-logistic|linear_svm"]
    assert np.allclose(ensemble[["prob_class_0", "prob_class_1"]].sum(axis=1), 1.0)


def test_ensemble_model_hash_depends_on_aligned_source_hashes() -> None:
    observations = _source_observations()
    observations["preprocessing_hash"] = observations["decoder"].map(
        {
            "logistic": "pre-logistic",
            "linear_svm": "pre-linear",
        }
    )
    observations["model_hash"] = observations.apply(
        lambda row: f"model-{row['decoder']}-{row['time']}",
        axis=1,
    )
    ensemble = ensemble_probability_observations(observations, baseline_window=(-0.25, -0.15))

    assert "source_model_hashes" in ensemble.columns
    assert "source_preprocessing_hashes" in ensemble.columns
    assert ensemble.groupby("time")["model_hash"].nunique().tolist() == [1, 1]
    assert ensemble.groupby("time")["preprocessing_hash"].nunique().tolist() == [1, 1]
    assert ensemble["model_hash"].nunique() == 2
    assert ensemble["source_model_hashes"].str.contains("logistic:model-logistic").all()
    assert ensemble["source_model_hashes"].str.contains("linear_svm:model-linear_svm").all()

    changed = observations.copy()
    changed.loc[changed["decoder"] == "linear_svm", "model_hash"] = (
        changed.loc[changed["decoder"] == "linear_svm", "model_hash"].astype(str) + "-changed"
    )
    changed_ensemble = ensemble_probability_observations(changed, baseline_window=(-0.25, -0.15))

    assert changed_ensemble["model_hash"].tolist() != ensemble["model_hash"].tolist()
    assert changed_ensemble["preprocessing_hash"].tolist() == ensemble["preprocessing_hash"].tolist()


def test_ensemble_probability_observations_accepts_integer_like_float_labels() -> None:
    observations = _source_observations()
    observations["true_label"] = observations["true_label"].astype(float)

    ensemble = ensemble_probability_observations(
        observations,
        baseline_window=(-0.25, -0.15),
    )

    assert ensemble["true_label"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert ensemble["is_correct"].tolist() == [True, False, True, True]


def test_ensemble_source_temperature_can_soften_overconfident_source() -> None:
    observations = pd.DataFrame(
        [
            {
                "subject": "sub-01",
                "fold": 0,
                "decoder": decoder,
                "emission_mode": "calibrated",
                "time": 0.1,
                "sample_index": 0,
                "sequence_id": 0,
                "true_label": 1,
                "true_class": "one",
                "class_0": "zero",
                "class_1": "one",
                "prob_class_0": prob_0,
                "prob_class_1": prob_1,
            }
            for decoder, prob_0, prob_1 in (
                ("overconfident_source", 0.99, 0.01),
                ("better_source", 0.20, 0.80),
            )
        ]
    )

    unscaled = ensemble_probability_observations(
        observations,
        decoders=("overconfident_source", "better_source"),
        baseline_window=None,
    )
    softened = ensemble_probability_observations(
        observations,
        decoders=("overconfident_source", "better_source"),
        baseline_window=None,
        source_temperatures=(10.0, 1.0),
    )

    assert unscaled["predicted_label"].tolist() == [0]
    assert softened["predicted_label"].tolist() == [1]
    assert softened["ensemble_source_temperatures"].unique().tolist() == ["10|1"]


def test_ensemble_probability_score_mode_can_rescue_geometric_overconfidence() -> None:
    observations = pd.DataFrame(
        [
            {
                "subject": "sub-01",
                "fold": 0,
                "decoder": decoder,
                "emission_mode": "calibrated",
                "time": 0.1,
                "sample_index": 0,
                "sequence_id": 0,
                "true_label": 1,
                "true_class": "one",
                "class_0": "zero",
                "class_1": "one",
                "prob_class_0": prob_0,
                "prob_class_1": prob_1,
            }
            for decoder, prob_0, prob_1 in (
                ("overconfident_source", 0.99, 0.01),
                ("better_source_a", 0.20, 0.80),
                ("better_source_b", 0.20, 0.80),
            )
        ]
    )

    geometric = ensemble_probability_observations(
        observations,
        decoders=("overconfident_source", "better_source_a", "better_source_b"),
        weights=(1.0, 1.0, 1.0),
        baseline_window=None,
    )
    probability_mean = ensemble_probability_observations(
        observations,
        decoders=("overconfident_source", "better_source_a", "better_source_b"),
        weights=(1.0, 1.0, 1.0),
        baseline_window=None,
        score_mode="probability",
    )
    rank_mean = ensemble_probability_observations(
        observations,
        decoders=("overconfident_source", "better_source_a", "better_source_b"),
        weights=(1.0, 1.0, 1.0),
        baseline_window=None,
        score_mode="rank",
    )

    assert geometric["predicted_label"].tolist() == [0]
    assert geometric["ensemble_score_mode"].unique().tolist() == ["log"]
    assert probability_mean["predicted_label"].tolist() == [1]
    assert probability_mean["ensemble_score_mode"].unique().tolist() == ["probability"]
    assert rank_mean["predicted_label"].tolist() == [1]
    assert rank_mean["ensemble_score_mode"].unique().tolist() == ["rank"]


def test_ensemble_confidence_probability_score_mode_downweights_uncertain_sources() -> None:
    observations = pd.DataFrame(
        [
            {
                "subject": "sub-01",
                "fold": 0,
                "decoder": decoder,
                "emission_mode": "calibrated",
                "time": 0.1,
                "sample_index": 0,
                "sequence_id": 0,
                "true_label": 1,
                "true_class": "one",
                "class_0": "zero",
                "class_1": "one",
                "prob_class_0": prob_0,
                "prob_class_1": prob_1,
            }
            for decoder, prob_0, prob_1 in (
                ("high_weight_uncertain_source", 0.55, 0.45),
                ("low_weight_confident_source", 0.05, 0.95),
            )
        ]
    )

    probability_mean = ensemble_probability_observations(
        observations,
        decoders=("high_weight_uncertain_source", "low_weight_confident_source"),
        weights=(10.0, 1.0),
        baseline_window=None,
        score_mode="probability",
    )
    confidence_weighted = ensemble_probability_observations(
        observations,
        decoders=("high_weight_uncertain_source", "low_weight_confident_source"),
        weights=(10.0, 1.0),
        baseline_window=None,
        score_mode="confidence_probability",
    )

    assert probability_mean["predicted_label"].tolist() == [0]
    assert confidence_weighted["predicted_label"].tolist() == [1]
    assert confidence_weighted["ensemble_score_mode"].unique().tolist() == ["confidence_probability"]


def test_ensemble_agreement_probability_score_mode_downweights_outlier_sources() -> None:
    observations = pd.DataFrame(
        [
            {
                "subject": "sub-01",
                "fold": 0,
                "decoder": decoder,
                "emission_mode": "calibrated",
                "time": 0.1,
                "sample_index": 0,
                "sequence_id": 0,
                "true_label": 1,
                "true_class": "one",
                "class_0": "zero",
                "class_1": "one",
                "prob_class_0": prob_0,
                "prob_class_1": prob_1,
            }
            for decoder, prob_0, prob_1 in (
                ("overweighted_outlier", 0.95, 0.05),
                ("agreeing_source_a", 0.25, 0.75),
                ("agreeing_source_b", 0.25, 0.75),
            )
        ]
    )

    probability_mean = ensemble_probability_observations(
        observations,
        decoders=("overweighted_outlier", "agreeing_source_a", "agreeing_source_b"),
        weights=(2.0, 1.0, 1.0),
        baseline_window=None,
        score_mode="probability",
    )
    agreement_weighted = ensemble_probability_observations(
        observations,
        decoders=("overweighted_outlier", "agreeing_source_a", "agreeing_source_b"),
        weights=(2.0, 1.0, 1.0),
        baseline_window=None,
        score_mode="agreement_probability",
    )

    assert probability_mean["predicted_label"].tolist() == [0]
    assert agreement_weighted["predicted_label"].tolist() == [1]
    assert agreement_weighted["ensemble_score_mode"].unique().tolist() == ["agreement_probability"]


def test_ensemble_source_baseline_debiasing_removes_source_level_bias() -> None:
    observations = pd.DataFrame(
        [
            {
                "subject": "sub-01",
                "fold": 0,
                "decoder": decoder,
                "emission_mode": "calibrated",
                "time": time,
                "sample_index": 0 if time < 0 else 1,
                "sequence_id": 0 if time < 0 else 1,
                "true_label": 1,
                "true_class": "one",
                "class_0": "zero",
                "class_1": "one",
                "prob_class_0": prob_0,
                "prob_class_1": 1.0 - prob_0,
            }
            for decoder, baseline_prob_0, effect_prob_0 in (
                ("biased_source", 0.93, 0.72),
                ("signal_source", 0.37, 0.18),
            )
            for time, prob_0 in ((-0.20, baseline_prob_0), (0.10, effect_prob_0))
        ]
    )

    plain = ensemble_probability_observations(
        observations,
        decoders=("biased_source", "signal_source"),
        weights=(0.9, 0.38),
        baseline_window=(-0.25, -0.15),
        score_mode="probability",
    )
    source_debiased = ensemble_probability_observations(
        observations,
        decoders=("biased_source", "signal_source"),
        weights=(0.9, 0.38),
        baseline_window=(-0.25, -0.15),
        score_mode="probability",
        source_baseline_debiasing=True,
    )

    plain_effect = plain.loc[plain["time"] == 0.10, "prob_class_1"].iloc[0]
    debiased_effect = source_debiased.loc[source_debiased["time"] == 0.10, "prob_class_1"].iloc[0]
    assert debiased_effect > plain_effect + 0.05
    assert source_debiased["source_baseline_debiasing"].unique().tolist() == [True]


def test_ensemble_probability_observations_rejects_fractional_true_labels() -> None:
    observations = _source_observations()
    observations["true_label"] = observations["true_label"].astype(float)
    observations.loc[observations["sequence_id"] == 0, "true_label"] = 0.5

    with pytest.raises(ValueError, match="true_label values must be integer-valued class labels"):
        ensemble_probability_observations(
            observations,
            baseline_window=(-0.25, -0.15),
        )


def test_summarize_ensemble_metrics_returns_time_resolved_rows() -> None:
    ensemble = ensemble_probability_observations(
        _source_observations(),
        baseline_window=(-0.25, -0.15),
    )

    metrics = summarize_ensemble_metrics(ensemble)

    assert metrics["time"].tolist() == [-0.20, 0.10]
    assert metrics["accuracy"].tolist() == [0.5, 1.0]
    assert metrics["n_test"].tolist() == [2, 2]
    assert metrics["class_names"].tolist() == ["zero|one", "zero|one"]


def test_summarize_ensemble_metrics_preserves_alignment_provenance() -> None:
    observations = _source_observations()
    observations["alignment_method"] = "procrustes"
    observations["alignment_times"] = "same_decode_window"
    observations["alignment_target_projection"] = "oracle_target_calibrated_alignment"
    observations["alignment_oracle_target_calibrated"] = True
    observations["alignment_debug_upper_bound"] = True
    observations["alignment_valid_for_benchmark"] = False
    observations["alignment_protocol_note"] = "debug upper bound only; not valid for benchmark"
    ensemble = ensemble_probability_observations(observations, baseline_window=(-0.25, -0.15))

    metrics = summarize_ensemble_metrics(ensemble)

    assert metrics["alignment_method"].unique().tolist() == ["procrustes"]
    assert metrics["alignment_times"].unique().tolist() == ["same_decode_window"]
    assert metrics["alignment_target_projection"].unique().tolist() == ["oracle_target_calibrated_alignment"]
    assert metrics["alignment_oracle_target_calibrated"].unique().tolist() == [True]
    assert metrics["alignment_debug_upper_bound"].unique().tolist() == [True]
    assert metrics["alignment_valid_for_benchmark"].unique().tolist() == [False]
    assert metrics["alignment_protocol_note"].unique().tolist() == ["debug upper bound only; not valid for benchmark"]


def test_summarize_ensemble_metrics_rejects_fractional_true_labels() -> None:
    ensemble = ensemble_probability_observations(
        _source_observations(),
        baseline_window=(-0.25, -0.15),
    )
    ensemble["true_label"] = ensemble["true_label"].astype(float)
    ensemble.loc[ensemble["time"] == -0.20, "true_label"] = 0.5

    with pytest.raises(ValueError, match="true_label values must be integer-valued class labels"):
        summarize_ensemble_metrics(ensemble)


def test_ensemble_cli_writes_observations_and_metrics(tmp_path: Path) -> None:
    source_path = tmp_path / "source_observations.csv"
    ensemble_path = tmp_path / "ensemble_observations.csv"
    metrics_path = tmp_path / "ensemble_metrics.csv"
    _source_observations().to_csv(source_path, index=False)

    exit_code = main(
        [
            str(source_path),
            "--out",
            str(ensemble_path),
            "--metrics-out",
            str(metrics_path),
            "--baseline-window",
            "-0.25",
            "-0.15",
        ]
    )

    assert exit_code == 0
    ensemble = pd.read_csv(ensemble_path)
    metrics = pd.read_csv(metrics_path)
    assert ensemble["decoder"].unique().tolist() == [DEFAULT_ENSEMBLE_DECODER]
    assert metrics["accuracy"].tolist() == [0.5, 1.0]


def test_ensemble_rejects_misaligned_source_rows() -> None:
    observations = _source_observations()
    misaligned = observations.drop(observations.loc[observations["decoder"] == "linear_svm"].index[-1])

    with pytest.raises(ValueError, match="does not align one-to-one"):
        ensemble_probability_observations(
            misaligned,
            baseline_window=(-0.25, -0.15),
        )
