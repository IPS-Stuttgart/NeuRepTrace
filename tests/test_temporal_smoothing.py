from pathlib import Path

import pandas as pd
import pytest

from neureptrace.temporal_smoothing import metrics_from_probability_observations, smooth_probability_observations


def _noisy_observation_frame() -> pd.DataFrame:
    rows = []
    for sequence_id in range(16):
        for time, p0 in [(0.10, 0.92), (0.20, 0.88), (0.30, 0.42), (0.40, 0.86), (0.50, 0.90)]:
            p1 = 1.0 - p0
            predicted_label = 0 if p0 >= p1 else 1
            rows.append(
                {
                    "subject": "sub-01",
                    "fold": sequence_id % 4,
                    "split_id": "stratified-kfold-4",
                    "seed": 13,
                    "decoder": "logistic",
                    "backend": "sklearn",
                    "emission_mode": "calibrated",
                    "feature_preprocessor": "pca_whiten",
                    "pca_components": "0.95",
                    "tuned_hyperparameters": True,
                    "tuning_cv_splits": 2,
                    "tuning_scoring": "balanced_accuracy",
                    "tuning_c_grid": "0.1|1.0",
                    "temporal_mode": "same_time",
                    "train_time": time,
                    "test_time": time,
                    "time": time,
                    "window_start": time - 0.01,
                    "window_stop": time + 0.01,
                    "sample_index": sequence_id,
                    "sequence_id": sequence_id,
                    "session": "ses-01",
                    "true_label": 0,
                    "true_class": "left",
                    "predicted_label": predicted_label,
                    "predicted_class": "left" if predicted_label == 0 else "right",
                    "probability_true_class": p0,
                    "confidence": max(p0, p1),
                    "is_correct": predicted_label == 0,
                    "class_0": "left",
                    "class_1": "right",
                    "prob_class_0": p0,
                    "prob_class_1": p1,
                    "model_hash": "base-model",
                }
            )
    return pd.DataFrame(rows)


def _nonzero_label_observation_frame() -> pd.DataFrame:
    observations = _noisy_observation_frame()
    label_map = {0: 10, 1: 20}
    observations["true_label"] = observations["true_label"].map(label_map)
    observations["predicted_label"] = observations["predicted_label"].map(label_map)
    observations["class_10"] = observations.pop("class_0")
    observations["class_20"] = observations.pop("class_1")
    observations = observations.rename(columns={"prob_class_0": "prob_class_10", "prob_class_1": "prob_class_20"})
    observations["probability_true_class"] = observations["prob_class_10"]
    observations["predicted_class"] = observations["predicted_label"].map({10: "left", 20: "right"})
    observations["is_correct"] = observations["predicted_label"] == observations["true_label"]
    return observations


def test_temporal_smoothing_exports_posteriors_and_metrics(tmp_path: Path):
    csv_path = tmp_path / "observations.csv"
    out_observations = tmp_path / "smoothed_observations.csv"
    out_metrics = tmp_path / "smoothed_metrics.csv"
    observations = _noisy_observation_frame()
    observations["label_shuffle_control"] = True
    observations["label_shuffle_seed"] = 13
    observations["alignment_method"] = "mcca"
    observations["alignment_valid_for_benchmark"] = False
    observations.to_csv(csv_path, index=False)

    smoothed, metrics = smooth_probability_observations(
        [csv_path],
        fit_window=(0.1, 0.5),
        stay_grid_size=40,
        out_observations=out_observations,
        out_metrics=out_metrics,
    )

    noisy_time = smoothed.loc[smoothed["time"].eq(0.30)]
    assert out_observations.exists()
    assert out_metrics.exists()
    assert smoothed[["prob_class_0", "prob_class_1"]].sum(axis=1).round(6).eq(1.0).all()
    assert smoothed["emission_mode"].unique().tolist() == ["calibrated_temporal_posterior"]
    assert smoothed["base_emission_mode"].unique().tolist() == ["calibrated"]
    assert smoothed["temporal_smoothing_method"].unique().tolist() == ["sticky_forward_backward"]
    assert noisy_time["prob_class_0"].min() > 0.5
    assert noisy_time["is_correct"].all()

    metric_row = metrics.loc[metrics["time"].eq(0.30)].iloc[0]
    assert metric_row["accuracy"] == 1.0
    assert metric_row["balanced_accuracy"] == 1.0
    assert metric_row["top2_accuracy"] == 1.0
    assert metric_row["top3_accuracy"] == 1.0
    assert metric_row["emission_mode"] == "calibrated_temporal_posterior"
    assert metric_row["feature_preprocessor"] == "pca_whiten"
    assert str(metric_row["tuned_hyperparameters"]).lower() == "true"
    assert metric_row["temporal_mode"] == "same_time"
    assert str(metric_row["label_shuffle_control"]).lower() == "true"
    assert str(metric_row["label_shuffle_seed"]) == "13"
    assert metric_row["alignment_method"] == "mcca"
    assert str(metric_row["alignment_valid_for_benchmark"]).lower() == "false"
    assert "temporal_smoothing_stay_probability" in metrics.columns


def test_temporal_smoothing_maps_nonzero_probability_labels(tmp_path: Path):
    csv_path = tmp_path / "observations.csv"
    observations = _nonzero_label_observation_frame()
    observations.to_csv(csv_path, index=False)

    smoothed, metrics = smooth_probability_observations(
        [csv_path],
        fit_window=(0.1, 0.5),
        stay_grid_size=40,
    )

    noisy_time = smoothed.loc[smoothed["time"].eq(0.30)]
    metric_row = metrics.loc[metrics["time"].eq(0.30)].iloc[0]

    assert smoothed["predicted_label"].unique().tolist() == [10]
    assert smoothed["predicted_class"].unique().tolist() == ["left"]
    assert noisy_time["probability_true_class"].min() > 0.5
    assert noisy_time["is_correct"].all()
    assert metric_row["accuracy"] == 1.0
    assert metric_row["top2_accuracy"] == 1.0
    assert metric_row["brier"] < 0.5
    assert metric_row["ece"] < 0.5


def test_temporal_smoothing_metrics_reject_fractional_true_labels() -> None:
    observations = _noisy_observation_frame()
    observations["true_label"] = observations["true_label"].astype(float)
    observations.loc[0, "true_label"] = 0.5

    with pytest.raises(ValueError, match="true_label values must be integer-valued"):
        metrics_from_probability_observations(observations)


def test_temporal_smoothing_metrics_reject_nonzero_labels_missing_from_probabilities() -> None:
    observations = _nonzero_label_observation_frame()
    observations.loc[0, "true_label"] = 30

    with pytest.raises(ValueError, match="true_label values must index prob_class_\\* labels"):
        metrics_from_probability_observations(observations)


def test_poststimulus_forward_smoothing_does_not_change_prestimulus_rows(tmp_path: Path):
    rows = []
    for sequence_id in range(12):
        for time, p0 in [(-0.10, 0.20), (0.04, 0.90), (0.18, 0.42), (0.30, 0.88), (0.42, 0.20)]:
            rows.append(
                {
                    "subject": "sub-01",
                    "fold": sequence_id % 3,
                    "decoder": "logistic",
                    "emission_mode": "calibrated",
                    "time": time,
                    "sample_index": sequence_id,
                    "sequence_id": sequence_id,
                    "true_label": 0,
                    "true_class": "left",
                    "predicted_label": 0 if p0 >= 0.5 else 1,
                    "predicted_class": "left" if p0 >= 0.5 else "right",
                    "probability_true_class": p0,
                    "confidence": max(p0, 1.0 - p0),
                    "is_correct": p0 >= 0.5,
                    "class_0": "left",
                    "class_1": "right",
                    "prob_class_0": p0,
                    "prob_class_1": 1.0 - p0,
                    "model_hash": "base-model",
                }
            )
    csv_path = tmp_path / "observations.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    smoothed, metrics = smooth_probability_observations(
        [csv_path],
        fit_window=(0.04, 0.30),
        stay_grid_size=40,
        mode="poststimulus_forward_only",
        apply_window=(0.04, 0.30),
    )

    prestim = smoothed.loc[smoothed["time"].eq(-0.10)]
    noisy_poststim = smoothed.loc[smoothed["time"].eq(0.18)]
    late = smoothed.loc[smoothed["time"].eq(0.42)]

    assert prestim["prob_class_0"].round(6).unique().tolist() == [0.2]
    assert late["prob_class_0"].round(6).unique().tolist() == [0.2]
    assert noisy_poststim["prob_class_0"].min() > 0.5
    assert smoothed["temporal_smoothing_method"].unique().tolist() == ["sticky_poststimulus_forward_only"]
    assert smoothed["temporal_smoothing_apply_window_start"].unique().tolist() == [0.04]
    assert smoothed["temporal_smoothing_apply_window_stop"].unique().tolist() == [0.3]

    prestim_metric = metrics.loc[metrics["time"].eq(-0.10)].iloc[0]
    poststim_metric = metrics.loc[metrics["time"].eq(0.18)].iloc[0]
    assert prestim_metric["accuracy"] == 0.0
    assert poststim_metric["accuracy"] == 1.0
