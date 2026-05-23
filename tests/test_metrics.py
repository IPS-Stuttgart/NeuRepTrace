import numpy as np
import pandas as pd
import pytest

from neureptrace.metrics import (
    brier_score_multiclass,
    compare_prepost_windows,
    confusion_category_enrichment,
    confusion_category_matrix,
    confusion_counts,
    confusion_pair_summary,
    expected_calibration_error,
    negative_log_likelihood,
    per_class_accuracy,
    reliability_bins,
    summarize_window_metric,
    top_k_accuracy,
    validate_probability_inputs,
)


def test_expected_calibration_error_perfect_predictions_are_zero():
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    labels = np.array([0, 1, 0])

    assert expected_calibration_error(probabilities, labels) == 0.0


def test_brier_score_multiclass_perfect_predictions_are_zero():
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])

    assert brier_score_multiclass(probabilities, labels) == 0.0


def test_negative_log_likelihood_scores_true_class_probabilities():
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.3, 0.7]])
    labels = np.array([0, 1, 1])

    expected = -np.mean(np.log([0.8, 0.9, 0.7]))
    assert negative_log_likelihood(probabilities, labels) == pytest.approx(expected)


def test_top_k_accuracy_handles_multiclass_predictions():
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.4, 0.35, 0.25],
            [0.1, 0.6, 0.3],
        ]
    )
    labels = np.array([0, 2, 1])

    assert top_k_accuracy(probabilities, labels, k=1) == pytest.approx(2 / 3)
    assert top_k_accuracy(probabilities, labels, k=2) == 1.0


def test_probability_metrics_reject_invalid_probability_rows():
    labels = np.array([0])

    with pytest.raises(ValueError, match="sum to one"):
        expected_calibration_error(np.array([[0.2, 0.2]]), labels)

    with pytest.raises(ValueError, match="non-negative"):
        brier_score_multiclass(np.array([[1.1, -0.1]]), labels)

    with pytest.raises(ValueError, match="valid column indices"):
        negative_log_likelihood(np.array([[0.5, 0.5]]), np.array([2]))


def test_validate_probability_inputs_can_accept_unnormalized_scores_when_requested():
    scores = np.array([[2.0, 1.0], [0.5, 0.25]])

    probabilities, labels = validate_probability_inputs(scores, require_normalized=False)

    assert labels is None
    np.testing.assert_allclose(probabilities, scores)


def test_reliability_bins_reports_confidence_accuracy_gap():
    probabilities = np.array([[0.8, 0.2], [0.7, 0.3], [0.4, 0.6]])
    labels = np.array([0, 1, 1])

    bins = reliability_bins(probabilities, labels, n_bins=2)

    assert bins[0]["n_samples"] == 0
    assert bins[1]["n_samples"] == 3
    assert round(float(bins[1]["accuracy"]), 3) == 0.667
    assert round(float(bins[1]["confidence"]), 3) == 0.700
    assert round(float(bins[1]["gap"]), 3) == -0.033


def test_summarize_window_metric_groups_inclusive_time_window():
    frame = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic", "logistic", "svm"],
            "time": [-0.2, 0.0, 0.2, 0.0],
            "accuracy": [0.4, 0.6, 0.8, 0.7],
        }
    )

    summary = summarize_window_metric(frame, "accuracy", (-0.2, 0.0), group_columns=("decoder",))

    logistic = summary.loc[summary["decoder"] == "logistic"].iloc[0]
    assert logistic["n_rows"] == 2
    assert logistic["accuracy_mean"] == 0.5
    assert logistic["window_start"] == -0.2
    assert logistic["window_stop"] == 0.0


def test_compare_prepost_windows_reports_grouped_delta():
    frame = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic", "logistic", "logistic", "svm", "svm"],
            "time": [-0.2, 0.0, 0.2, 0.4, -0.2, 0.2],
            "accuracy": [0.4, 0.6, 0.8, 0.9, 0.5, 0.75],
        }
    )

    comparison = compare_prepost_windows(frame, "accuracy", (-0.2, 0.0), (0.2, 0.4), group_columns=("decoder",))

    logistic = comparison.loc[comparison["decoder"] == "logistic"].iloc[0]
    assert logistic["n_pre_rows"] == 2
    assert logistic["n_post_rows"] == 2
    assert round(float(logistic["accuracy_post_minus_pre"]), 3) == 0.35


def test_confusion_counts_accepts_pymegdec_style_columns():
    predictions = pd.DataFrame(
        {
            "participant": ["p1", "p1", "p1", "p2"],
            "window": [0.1, 0.1, 0.1, 0.1],
            "true_stimulus": ["cat", "cat", "dog", "cat"],
            "predicted_stimulus": ["cat", "dog", "dog", "cat"],
        }
    )

    counts = confusion_counts(
        predictions,
        true_column="true_stimulus",
        predicted_column="predicted_stimulus",
        group_columns=("window",),
    )

    cat_as_cat = counts[(counts["true_label"] == "cat") & (counts["predicted_label"] == "cat")].iloc[0]
    assert cat_as_cat["count"] == 2


def test_per_class_accuracy_counts_participants():
    predictions = pd.DataFrame(
        {
            "participant": ["p1", "p1", "p1", "p2"],
            "task": ["animate", "animate", "animate", "animate"],
            "true_label": ["animal", "animal", "device", "animal"],
            "predicted_label": ["animal", "device", "device", "animal"],
        }
    )

    summary = per_class_accuracy(predictions, participant_column="participant", group_columns=("task",))

    animal = summary.loc[summary["true_label"] == "animal"].iloc[0]
    assert animal["n_trials"] == 3
    assert animal["n_correct"] == 2
    assert round(float(animal["accuracy"]), 3) == 0.667
    assert animal["n_participants"] == 2


def test_confusion_pair_summary_reports_bidirectional_lift_and_metadata():
    predictions = pd.DataFrame(
        {
            "participant": ["p1", "p1", "p2", "p2", "p1", "p2"],
            "decoder": ["logistic"] * 6,
            "true_stimulus": [1, 1, 2, 1, 3, 4],
            "predicted_stimulus": [2, 2, 1, 1, 4, 3],
        }
    )
    metadata = pd.DataFrame(
        {
            "stimulus": [1, 2, 3, 4],
            "name": ["cat", "dog", "cup", "bottle"],
            "semantic_category": ["animal", "animal", "object", "object"],
        }
    )

    pairs = confusion_pair_summary(
        predictions,
        true_column="true_stimulus",
        predicted_column="predicted_stimulus",
        group_columns=("decoder",),
        participant_column="participant",
        metadata_frame=metadata,
        label_prefix="stimulus",
    )

    pair = pairs[(pairs["stimulus_a"] == 1) & (pairs["stimulus_b"] == 2)].iloc[0]
    assert pair["a_to_b_count"] == 2
    assert pair["b_to_a_count"] == 1
    assert pair["total_confusions"] == 3
    assert pair["n_confused_participants"] == 2
    assert bool(pair["same_semantic_category"]) is True
    assert pair["pair_confusion_lift"] > 1.0


def test_confusion_category_enrichment_and_matrix_use_error_marginals():
    predictions = pd.DataFrame(
        {
            "participant": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "decoder": ["logistic"] * 6,
            "true_label": [1, 1, 2, 1, 3, 4],
            "predicted_label": [2, 2, 1, 1, 4, 3],
        }
    )
    metadata = pd.DataFrame(
        {
            "label": [1, 2, 3, 4],
            "semantic_category": ["animal", "animal", "object", "object"],
        }
    )

    enrichment = confusion_category_enrichment(
        predictions,
        metadata_frame=metadata,
        category_columns=("semantic_category",),
        group_columns=("decoder",),
        participant_column="participant",
        n_permutations=0,
    )
    row = enrichment.iloc[0]
    assert row["category_column"] == "semantic_category"
    assert row["n_errors_with_category"] == 5
    assert row["same_category_errors"] == 5
    assert row["expected_same_category_errors"] == 2.6
    assert row["same_category_lift"] > 1.0
    assert row["n_participants_with_same_category_errors"] == 3

    matrix = confusion_category_matrix(
        predictions,
        metadata_frame=metadata,
        category_columns=("semantic_category",),
        group_columns=("decoder",),
        participant_column="participant",
    )
    animal = matrix[(matrix["true_category"] == "animal") & (matrix["predicted_category"] == "animal")].iloc[0]
    assert animal["count"] == 3
    assert animal["expected_count"] == 1.8
    assert animal["category_confusion_lift"] > 1.0
