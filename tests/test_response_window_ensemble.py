from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.response_window_ensemble import run_response_window_ensemble


def _toy_observations() -> pd.DataFrame:
    rows = []
    times = (0.088, 0.136, 0.184, 0.232, 0.280)
    for subject in ("sub-01", "sub-02", "sub-03"):
        for sample_index, true_label in enumerate((0, 1, 2, 0, 1, 2)):
            for time_index, time in enumerate(times):
                probabilities = np.full(3, 0.1)
                if time_index == 2:
                    probabilities[true_label] = 0.8
                elif subject == "sub-03" and time_index == 0:
                    probabilities[true_label] = 0.7
                else:
                    probabilities[(true_label + 1) % 3] = 0.8
                probabilities = probabilities / probabilities.sum()
                predicted_label = int(probabilities.argmax())
                rows.append(
                    {
                        "subject": subject,
                        "fold": subject,
                        "decoder": "base",
                        "emission_mode": "calibrated",
                        "time": time,
                        "test_time": time,
                        "sample_index": sample_index,
                        "sequence_id": sample_index,
                        "true_label": true_label,
                        "true_class": f"class-{true_label}",
                        "predicted_label": predicted_label,
                        "predicted_class": f"class-{predicted_label}",
                        "probability_true_class": float(probabilities[true_label]),
                        "confidence": float(probabilities.max()),
                        "class_0": "class-0",
                        "class_1": "class-1",
                        "class_2": "class-2",
                        "prob_class_0": float(probabilities[0]),
                        "prob_class_1": float(probabilities[1]),
                        "prob_class_2": float(probabilities[2]),
                    }
                )
    return pd.DataFrame(rows)


def _toy_decoder_observations() -> pd.DataFrame:
    rows = []
    times = (0.088, 0.136)
    for subject in ("sub-01", "sub-02", "sub-03"):
        for sample_index, true_label in enumerate((0, 1, 2, 0, 1, 2)):
            for decoder in ("weak", "strong"):
                for time in times:
                    probabilities = np.full(3, 0.1)
                    if decoder == "strong":
                        probabilities[true_label] = 0.8
                    else:
                        probabilities[(true_label + 1) % 3] = 0.8
                    probabilities = probabilities / probabilities.sum()
                    predicted_label = int(probabilities.argmax())
                    rows.append(
                        {
                            "subject": subject,
                            "fold": subject,
                            "decoder": decoder,
                            "emission_mode": "calibrated",
                            "time": time,
                            "test_time": time,
                            "sample_index": sample_index,
                            "sequence_id": sample_index,
                            "true_label": true_label,
                            "true_class": f"class-{true_label}",
                            "predicted_label": predicted_label,
                            "predicted_class": f"class-{predicted_label}",
                            "probability_true_class": float(probabilities[true_label]),
                            "confidence": float(probabilities.max()),
                            "class_0": "class-0",
                            "class_1": "class-1",
                            "class_2": "class-2",
                            "prob_class_0": float(probabilities[0]),
                            "prob_class_1": float(probabilities[1]),
                            "prob_class_2": float(probabilities[2]),
                        }
                    )
    return pd.DataFrame(rows)


def test_response_window_uniform_logit_ensemble_writes_metrics(tmp_path: Path):
    csv_path = tmp_path / "observations.csv"
    _toy_observations().to_csv(csv_path, index=False)

    out_observations = tmp_path / "response_observations.csv"
    out_metrics = tmp_path / "response_metrics.csv"
    ensembled, metrics = run_response_window_ensemble(
        [csv_path],
        mode="uniform",
        out_observations=out_observations,
        out_metrics=out_metrics,
    )

    assert out_observations.exists()
    assert out_metrics.exists()
    assert ensembled["time"].unique().tolist() == [0.184]
    assert ensembled["response_window_mode"].unique().tolist() == ["uniform"]
    assert ensembled["response_window_actual_times"].unique().tolist() == ["0.088|0.136|0.184|0.232|0.28"]
    assert metrics["decoder"].unique().tolist() == ["poststimulus_response_window_logit_ensemble"]
    assert metrics["balanced_accuracy"].between(0.0, 1.0).all()


def test_response_window_rejects_duplicate_nearest_time_mapping(tmp_path: Path):
    csv_path = tmp_path / "observations.csv"
    observations = _toy_observations()
    observations = observations.loc[observations["time"] != 0.280]
    observations.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="collapse to duplicate decoded time centers"):
        run_response_window_ensemble([csv_path], mode="uniform")


def test_plain_response_window_rejects_multiple_decoders(tmp_path: Path):
    observations = pd.concat(
        [
            _toy_observations(),
            _toy_observations().assign(decoder="other"),
        ],
        ignore_index=True,
    )
    csv_path = tmp_path / "observations.csv"
    observations.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="multiple decoder values"):
        run_response_window_ensemble([csv_path], mode="uniform")


def test_plain_response_window_rejects_multiple_emission_modes(tmp_path: Path):
    observations = pd.concat(
        [
            _toy_observations(),
            _toy_observations().assign(emission_mode="uncalibrated"),
        ],
        ignore_index=True,
    )
    csv_path = tmp_path / "observations.csv"
    observations.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="multiple emission_mode values"):
        run_response_window_ensemble([csv_path], mode="uniform")


def test_plain_response_window_rejects_duplicate_trial_time_rows(tmp_path: Path):
    observations = _toy_observations()
    duplicate = observations.iloc[[0]].copy()
    observations = pd.concat([observations, duplicate], ignore_index=True)
    csv_path = tmp_path / "observations.csv"
    observations.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="duplicate rows"):
        run_response_window_ensemble([csv_path], mode="uniform")


def test_response_window_model_hash_depends_on_source_time_hashes(tmp_path: Path):
    observations = _toy_observations()
    observations["preprocessing_hash"] = observations["time"].map(lambda time: f"pre-{time:.3f}")
    observations["model_hash"] = observations["time"].map(lambda time: f"model-{time:.3f}")
    csv_path = tmp_path / "observations.csv"
    observations.to_csv(csv_path, index=False)

    ensembled, metrics = run_response_window_ensemble(
        [csv_path],
        response_times=(0.088, 0.184),
        mode="uniform",
    )

    assert "response_window_source_model_hashes" in ensembled.columns
    assert ensembled["response_window_source_model_hashes"].str.contains("0.088:model-0.088").all()
    assert ensembled["response_window_source_model_hashes"].str.contains("0.184:model-0.184").all()
    assert metrics["response_window_source_model_hashes"].unique().tolist() == [
        "0.088:model-0.088|0.184:model-0.184"
    ]

    changed = observations.copy()
    changed.loc[changed["time"] == 0.184, "model_hash"] = "model-0.184-changed"
    changed_path = tmp_path / "observations_changed.csv"
    changed.to_csv(changed_path, index=False)
    changed_ensembled, _ = run_response_window_ensemble(
        [changed_path],
        response_times=(0.088, 0.184),
        mode="uniform",
    )

    assert changed_ensembled["model_hash"].tolist() != ensembled["model_hash"].tolist()
    assert changed_ensembled["preprocessing_hash"].tolist() == ensembled["preprocessing_hash"].tolist()


def test_response_window_metrics_preserve_constant_alignment_provenance(tmp_path: Path):
    observations = _toy_observations()
    observations["alignment_method"] = "mcca"
    observations["alignment_anchor_mode"] = "event_code_mean"
    observations["alignment_anchor_column"] = "trigger"
    observations["alignment_target_projection"] = "oracle_target_calibrated_alignment"
    observations["alignment_oracle_target_calibrated"] = True
    observations["alignment_debug_upper_bound"] = True
    observations["alignment_valid_for_benchmark"] = False
    csv_path = tmp_path / "observations.csv"
    observations.to_csv(csv_path, index=False)

    _ensembled, metrics = run_response_window_ensemble([csv_path], mode="uniform")

    assert metrics["alignment_method"].unique().tolist() == ["mcca"]
    assert metrics["alignment_anchor_mode"].unique().tolist() == ["event_code_mean"]
    assert metrics["alignment_anchor_column"].unique().tolist() == ["trigger"]
    assert metrics["alignment_target_projection"].unique().tolist() == ["oracle_target_calibrated_alignment"]
    assert metrics["alignment_oracle_target_calibrated"].unique().tolist() == [True]
    assert metrics["alignment_debug_upper_bound"].unique().tolist() == [True]
    assert metrics["alignment_valid_for_benchmark"].unique().tolist() == [False]


def test_response_window_uniform_probability_mean_uses_arithmetic_average(tmp_path: Path):
    csv_path = tmp_path / "observations.csv"
    _toy_observations().to_csv(csv_path, index=False)

    ensembled, _ = run_response_window_ensemble(
        [csv_path],
        response_times=(0.088, 0.184),
        mode="uniform",
        combine="probability_mean",
    )

    row = ensembled.loc[(ensembled["subject"] == "sub-01") & (ensembled["sample_index"] == 0)].iloc[0]
    assert row["response_window_combine"] == "probability_mean"
    assert row["response_window_actual_times"] == "0.088|0.184"
    np.testing.assert_allclose(
        [row["prob_class_0"], row["prob_class_1"], row["prob_class_2"]],
        [0.45, 0.45, 0.10],
    )


def test_response_window_poststimulus_forward_smooths_then_averages_requested_times(tmp_path: Path):
    observations = _toy_observations()
    prestimulus = observations.loc[observations["time"] == 0.088].copy()
    prestimulus["time"] = -0.050
    prestimulus["test_time"] = -0.050
    observations = pd.concat([prestimulus, observations], ignore_index=True)
    csv_path = tmp_path / "observations.csv"
    observations.to_csv(csv_path, index=False)

    ensembled, metrics = run_response_window_ensemble(
        [csv_path],
        response_times=(0.088, 0.136, 0.184, 0.232),
        mode="response_window_poststimulus_forward",
        smoothing_fit_window=(0.10, 0.30),
        smoothing_apply_window=(0.088, 0.232),
        smoothing_stay_grid_size=20,
    )

    assert ensembled["time"].unique().tolist() == [0.184]
    assert ensembled["response_window_mode"].unique().tolist() == ["response_window_poststimulus_forward"]
    assert ensembled["temporal_smoothing_method"].unique().tolist() == ["sticky_poststimulus_forward_only"]
    assert ensembled["temporal_smoothing_apply_window_start"].unique().tolist() == [0.088]
    assert ensembled["temporal_smoothing_apply_window_stop"].unique().tolist() == [0.232]
    assert ensembled["response_window_actual_times"].unique().tolist() == ["0.088|0.136|0.184|0.232"]
    assert metrics["balanced_accuracy"].between(0.0, 1.0).all()


def test_response_window_learned_weights_are_source_subject_only(tmp_path: Path):
    csv_path = tmp_path / "observations.csv"
    _toy_observations().to_csv(csv_path, index=False)

    ensembled, _ = run_response_window_ensemble(
        [csv_path],
        mode="source_oof_nonnegative",
        weight_grid_step=0.5,
    )

    weights = ensembled.groupby("subject")["response_window_weights"].first().to_dict()
    assert set(weights) == {"sub-01", "sub-02", "sub-03"}
    assert all(weight for weight in weights.values())
    assert ensembled["response_window_source_score"].replace("", np.nan).notna().all()


def test_response_window_can_learn_decoder_family_weights_from_source_subjects(tmp_path: Path):
    csv_path = tmp_path / "decoder_observations.csv"
    _toy_decoder_observations().to_csv(csv_path, index=False)

    ensembled, metrics = run_response_window_ensemble(
        [csv_path],
        response_times=(0.088, 0.136),
        mode="decoder_source_oof_nonnegative",
        weight_grid_step=0.5,
    )

    assert ensembled["response_window_mode"].unique().tolist() == ["decoder_source_oof_nonnegative"]
    assert ensembled["response_window_decoder_candidates"].unique().tolist() == ["weak|strong"]
    strong_weights = [
        float(weights.split("|")[1])
        for weights in ensembled.groupby("subject")["response_window_decoder_weights"].first()
    ]
    assert all(weight >= 0.5 for weight in strong_weights)
    assert ensembled["response_window_source_score"].replace("", np.nan).notna().all()
    assert metrics["balanced_accuracy"].between(0.0, 1.0).all()


def test_decoder_response_window_model_hash_depends_on_source_hashes(tmp_path: Path):
    observations = _toy_decoder_observations()
    observations["preprocessing_hash"] = observations["decoder"].map(lambda decoder: f"pre-{decoder}")
    observations["model_hash"] = observations.apply(
        lambda row: f"model-{row['decoder']}-{row['time']:.3f}",
        axis=1,
    )
    csv_path = tmp_path / "decoder_observations.csv"
    observations.to_csv(csv_path, index=False)

    ensembled, _metrics = run_response_window_ensemble(
        [csv_path],
        response_times=(0.088, 0.136),
        mode="decoder_source_oof_nonnegative",
        weight_grid_step=0.5,
    )

    assert ensembled["response_window_source_model_hashes"].str.contains("weak@0.088:model-weak-0.088").all()
    assert ensembled["response_window_source_model_hashes"].str.contains("strong@0.136:model-strong-0.136").all()

    changed = observations.copy()
    changed.loc[
        (changed["decoder"] == "strong") & (changed["time"] == 0.136),
        "model_hash",
    ] = "model-strong-0.136-changed"
    changed_path = tmp_path / "decoder_observations_changed.csv"
    changed.to_csv(changed_path, index=False)
    changed_ensembled, _ = run_response_window_ensemble(
        [changed_path],
        response_times=(0.088, 0.136),
        mode="decoder_source_oof_nonnegative",
        weight_grid_step=0.5,
    )

    assert changed_ensembled["model_hash"].tolist() != ensembled["model_hash"].tolist()
    assert changed_ensembled["preprocessing_hash"].tolist() == ensembled["preprocessing_hash"].tolist()


def test_decoder_response_window_rejects_duplicate_trial_time_rows(tmp_path: Path):
    observations = _toy_decoder_observations()
    duplicate = observations.iloc[[0]].copy()
    observations = pd.concat([observations, duplicate], ignore_index=True)
    csv_path = tmp_path / "decoder_observations.csv"
    observations.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="duplicate rows"):
        run_response_window_ensemble(
            [csv_path],
            response_times=(0.088, 0.136),
            mode="decoder_source_oof_nonnegative",
        )


def test_response_window_uses_outer_test_group_when_subject_is_empty(tmp_path: Path):
    observations = _toy_observations()
    observations["outer_test_groups"] = observations["subject"]
    observations["group"] = observations["subject"]
    observations["subject"] = np.nan
    csv_path = tmp_path / "observations.csv"
    observations.to_csv(csv_path, index=False)

    ensembled, _ = run_response_window_ensemble(
        [csv_path],
        mode="source_oof_nonnegative",
        weight_grid_step=0.5,
    )

    weights = ensembled.groupby("outer_test_groups")["response_window_weights"].first().to_dict()
    assert set(weights) == {"sub-01", "sub-02", "sub-03"}
    assert ensembled["subject"].isna().all()
    assert ensembled["response_window_source_score"].replace("", np.nan).notna().all()


def test_response_window_uses_session_when_subject_and_group_columns_are_empty(tmp_path: Path):
    observations = _toy_observations()
    observations["session"] = observations["subject"]
    observations["subject"] = np.nan
    observations["group"] = ""
    observations["outer_test_groups"] = np.nan
    csv_path = tmp_path / "observations.csv"
    observations.to_csv(csv_path, index=False)

    ensembled, _ = run_response_window_ensemble(
        [csv_path],
        mode="source_oof_nonnegative",
        weight_grid_step=0.5,
    )

    weights = ensembled.groupby("session")["response_window_weights"].first().to_dict()
    assert set(weights) == {"sub-01", "sub-02", "sub-03"}
    assert ensembled["response_window_source_score"].replace("", np.nan).notna().all()


def test_response_window_can_use_fold_as_last_resort_target_key(tmp_path: Path):
    observations = _toy_observations()
    observations["subject"] = np.nan
    observations["group"] = np.nan
    observations["outer_test_groups"] = np.nan
    csv_path = tmp_path / "observations.csv"
    observations.to_csv(csv_path, index=False)

    ensembled, _ = run_response_window_ensemble(
        [csv_path],
        mode="source_oof_nonnegative",
        weight_grid_step=0.5,
    )

    weights = ensembled.groupby("fold")["response_window_weights"].first().to_dict()
    assert set(weights) == {"sub-01", "sub-02", "sub-03"}
    assert ensembled["response_window_source_score"].replace("", np.nan).notna().all()
