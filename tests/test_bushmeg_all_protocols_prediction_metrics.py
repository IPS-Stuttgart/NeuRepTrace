from __future__ import annotations

import pandas as pd
import pytest

import neureptrace.bushmeg_all_protocols as all_protocols


def test_prediction_metric_frame_uses_true_label_index_for_string_labels() -> None:
    predictions = pd.DataFrame(
        [
            {
                "outer_test_subject": "subj-1",
                "trial_index": 0,
                "true_label": "face",
                "true_label_index": 0,
                "predicted_label": "face",
                "predicted_label_index": 0,
                "prob_class_0": 0.9,
                "prob_class_1": 0.1,
            },
            {
                "outer_test_subject": "subj-1",
                "trial_index": 1,
                "true_label": "scrambled",
                "true_label_index": 1,
                "predicted_label": "scrambled",
                "predicted_label_index": 1,
                "prob_class_0": 0.2,
                "prob_class_1": 0.8,
            },
        ]
    )

    metrics = all_protocols._prediction_metric_frame(predictions)

    row = metrics.iloc[0]
    assert row["outer_test_subject"] == "subj-1"
    assert row["accuracy"] == pytest.approx(1.0)
    assert row["balanced_accuracy"] == pytest.approx(1.0)
    assert row["top2_accuracy"] == pytest.approx(1.0)
    assert row["top3_accuracy"] == pytest.approx(1.0)
    assert row["log_loss"] < 0.3


def test_prediction_metric_frame_can_map_class_columns_when_label_index_missing() -> None:
    predictions = pd.DataFrame(
        [
            {
                "outer_test_subject": "subj-1",
                "true_label": "left",
                "class_0": "left",
                "class_1": "right",
                "prob_class_0": 0.7,
                "prob_class_1": 0.3,
            },
            {
                "outer_test_subject": "subj-1",
                "true_label": "right",
                "class_0": "left",
                "class_1": "right",
                "prob_class_0": 0.1,
                "prob_class_1": 0.9,
            },
        ]
    )

    metrics = all_protocols._prediction_metric_frame(predictions)

    assert metrics.loc[0, "accuracy"] == pytest.approx(1.0)
    assert metrics.loc[0, "balanced_accuracy"] == pytest.approx(1.0)


def test_prediction_metric_frame_resolves_top_k_ties_to_exact_k_classes() -> None:
    predictions = pd.DataFrame(
        [
            {
                "outer_test_subject": "subj-1",
                "true_label_index": label,
                "prob_class_0": 0.25,
                "prob_class_1": 0.25,
                "prob_class_2": 0.25,
                "prob_class_3": 0.25,
            }
            for label in range(4)
        ]
    )

    metrics = all_protocols._prediction_metric_frame(predictions)

    row = metrics.iloc[0]
    assert row["top2_accuracy"] == pytest.approx(0.5)
    assert row["top3_accuracy"] == pytest.approx(0.75)


def test_prediction_metric_frame_uses_label_only_predictions_without_probabilities() -> None:
    predictions = pd.DataFrame(
        [
            {
                "outer_test_subject": "subj-1",
                "fold_index": 1,
                "true_label": "left",
                "predicted_label": "left",
            },
            {
                "outer_test_subject": "subj-1",
                "fold_index": 1,
                "true_label": "right",
                "predicted_label": "left",
            },
            {
                "outer_test_subject": "subj-1",
                "fold_index": 1,
                "true_label": "right",
                "predicted_label": "right",
            },
        ]
    )

    metrics = all_protocols._prediction_metric_frame(predictions)

    row = metrics.iloc[0]
    assert row["outer_test_subject"] == "subj-1"
    assert row["fold_index"] == 1
    assert row["accuracy"] == pytest.approx(2.0 / 3.0)
    assert row["balanced_accuracy"] == pytest.approx(0.75)
    assert pd.isna(row["top2_accuracy"])
    assert pd.isna(row["log_loss"])


def test_prediction_metric_frame_keeps_fold_local_groups_for_repeated_subjects() -> None:
    predictions = pd.DataFrame(
        [
            {
                "outer_test_subject": "subj-1",
                "fold_index": 1,
                "true_label_index": 0,
                "prob_class_0": 0.9,
                "prob_class_1": 0.1,
            },
            {
                "outer_test_subject": "subj-1",
                "fold_index": 2,
                "true_label_index": 0,
                "prob_class_0": 0.1,
                "prob_class_1": 0.9,
            },
        ]
    )

    metrics = all_protocols._prediction_metric_frame(predictions).sort_values("fold_index").reset_index(drop=True)

    assert metrics["outer_test_subject"].tolist() == ["subj-1", "subj-1"]
    assert metrics["fold_index"].tolist() == [1, 2]
    assert metrics["accuracy"].tolist() == pytest.approx([1.0, 0.0])


def test_normalize_summary_merges_prediction_metrics_by_fold_not_subject_only() -> None:
    spec = all_protocols.MethodSpec(
        "few_shot_target_calibrated_decoder_k1",
        "few_shot",
        3,
        "protocol3_few_shot",
    )
    raw_summary = pd.DataFrame(
        [
            {
                "outer_test_subject": "subj-1",
                "fold_index": 1,
                "n_test_trials": 1,
                "n_train": 4,
                "n_train_subjects": 2,
                "accuracy": pd.NA,
                "balanced_accuracy": pd.NA,
            },
            {
                "outer_test_subject": "subj-1",
                "fold_index": 2,
                "n_test_trials": 1,
                "n_train": 4,
                "n_train_subjects": 2,
                "accuracy": pd.NA,
                "balanced_accuracy": pd.NA,
            },
        ]
    )
    raw_predictions = pd.DataFrame(
        [
            {
                "outer_test_subject": "subj-1",
                "fold_index": 1,
                "true_label_index": 0,
                "prob_class_0": 0.9,
                "prob_class_1": 0.1,
            },
            {
                "outer_test_subject": "subj-1",
                "fold_index": 2,
                "true_label_index": 0,
                "prob_class_0": 0.1,
                "prob_class_1": 0.9,
            },
        ]
    )

    normalized = all_protocols._normalize_summary(
        raw_summary,
        raw_predictions,
        spec=spec,
        config={},
    ).sort_values("fold_index").reset_index(drop=True)

    assert len(normalized) == 2
    assert normalized["accuracy"].tolist() == pytest.approx([1.0, 0.0])
    assert normalized["balanced_accuracy"].tolist() == pytest.approx([1.0, 0.0])
