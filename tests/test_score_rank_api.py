from __future__ import annotations

import numpy as np

from neureptrace.decoding import score_model_classes
from neureptrace.decoding.score_rank import (
    rank_summary_rows,
    score_window_classes,
    summarize_class_ranks,
    topk_rank_metrics,
    true_label_ranks,
)
from neureptrace.decoding.windowed import WindowedModelBundle


class BinaryDecisionModel:
    classes_ = np.asarray([0, 1])

    def decision_function(self, features):
        return np.asarray([1.0, -2.0, 0.5])[: np.asarray(features).shape[0]]


class PredictOnlyModel:
    classes_ = np.asarray([1, 2])

    def predict(self, features):
        return np.asarray([1, 2, 2])[: np.asarray(features).shape[0]]


def test_score_model_classes_expands_binary_decision_function():
    features = np.zeros((3, 2), dtype=float)

    scores, classes = score_model_classes(BinaryDecisionModel(), features)

    assert np.array_equal(classes, np.asarray([0, 1]))
    assert scores.shape == (3, 2)
    assert np.allclose(scores[:, 1], np.asarray([1.0, -2.0, 0.5]))
    assert np.allclose(scores[:, 0], -scores[:, 1])


def test_score_model_classes_can_return_empty_matrix_on_missing_scores():
    features = np.zeros((3, 2), dtype=float)

    scores, classes = score_model_classes(PredictOnlyModel(), features, empty_on_missing=True)

    assert scores.shape == (3, 0)
    assert np.all(np.isnan(scores))
    assert classes.shape == (0,)


def test_score_model_classes_predict_fallback_returns_one_hot_scores():
    features = np.zeros((3, 2), dtype=float)

    scores, classes = score_model_classes(PredictOnlyModel(), features, predict_fallback=True)

    assert np.array_equal(classes, np.asarray([1, 2]))
    assert np.array_equal(
        scores,
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        ),
    )


def test_score_window_classes_applies_window_bundle_scoring():
    bundle = WindowedModelBundle(
        model=BinaryDecisionModel(),
        train_window=(0.1, 0.2),
        train_labels=np.asarray([0, 1, 1]),
        pca_coeff=None,
        train_features_mean=None,
        explained_variance_percent=np.nan,
        actual_components_pca=2,
    )

    scores, classes = score_window_classes(bundle, np.zeros((3, 2), dtype=float))

    assert np.array_equal(classes, np.asarray([0, 1]))
    assert scores.shape == (3, 2)


def test_summarize_class_ranks_counts_missing_labels_as_topk_failures():
    scores = np.asarray(
        [
            [0.1, 0.9, 0.0],
            [0.9, 0.1, 0.0],
            [0.2, 0.3, 0.4],
        ]
    )
    classes = np.asarray([1, 2, 3])
    y_true = np.asarray([2, 3, 4])

    summary = summarize_class_ranks(y_true, scores, classes, top_k=(1, 2, 3), row_top_k=2, class_column="stimulus")

    assert summary["top_k_accuracy"][1] == 1.0 / 3.0
    assert summary["top_k_accuracy"][2] == 1.0 / 3.0
    assert summary["top_k_accuracy"][3] == 2.0 / 3.0
    assert np.allclose(summary["true_label_ranks"], np.asarray([1.0, 3.0, np.nan]), equal_nan=True)
    assert summary["mean_true_label_rank"] == 2.0

    rows = rank_summary_rows(y_true, scores, classes, row_top_k=2, class_column="stimulus")
    assert rows[0]["rank1_stimulus"] == 2
    assert rows[0]["true_label_rank"] == 1.0

    assert np.allclose(true_label_ranks(y_true, scores, classes), np.asarray([1.0, 3.0, np.nan]), equal_nan=True)

    metrics = topk_rank_metrics(y_true, scores, classes, top_k=(2, 3), include_true_label_ranks=True, include_median=True)
    assert metrics["top2_accuracy"] == 1.0 / 3.0
    assert metrics["top3_accuracy"] == 2.0 / 3.0
    assert metrics["mean_true_label_rank"] == 2.0
    assert metrics["median_true_label_rank"] == 2.0
    assert np.allclose(metrics["true_label_ranks"], np.asarray([1.0, 3.0, np.nan]), equal_nan=True)
