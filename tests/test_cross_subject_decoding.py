from __future__ import annotations

import numpy as np

from neureptrace.decoding.cross_subject import (
    DecoderCandidate,
    ParticipantFeatureSet,
    leave_one_subject_out,
    nested_leave_one_subject_out,
    rank_true_labels,
    summarize_cross_subject_folds,
)


class NearestMeanClassifier:
    def fit(self, features, labels):
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels)
        self.classes_ = np.unique(labels)
        self.centroids_ = np.vstack([np.mean(features[labels == label], axis=0) for label in self.classes_])
        return self

    def decision_function(self, features):
        features = np.asarray(features, dtype=float)
        differences = features[:, None, :] - self.centroids_[None, :, :]
        return -np.sum(differences * differences, axis=2)

    def predict(self, features):
        return self.classes_[np.argmax(self.decision_function(features), axis=1)]


class ConstantZeroClassifier:
    classes_ = np.array([0, 1])

    def predict(self, features):
        return np.zeros(np.asarray(features).shape[0], dtype=int)

    def decision_function(self, features):
        features = np.asarray(features)
        return np.column_stack([np.ones(features.shape[0]), np.zeros(features.shape[0])])


def _fit_nearest_mean(features, labels):
    return NearestMeanClassifier().fit(features, labels)


def _fit_constant_zero(_features, _labels):
    return ConstantZeroClassifier()


def _feature_sets():
    base_features = np.array(
        [
            [-2.0, -1.8],
            [-1.7, -2.1],
            [1.8, 2.0],
            [2.1, 1.7],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    return tuple(
        ParticipantFeatureSet(
            participant=f"s{participant}",
            features=base_features + 0.01 * participant,
            labels=labels,
            sample_ids=np.arange(10, 14) + 100 * participant,
        )
        for participant in range(3)
    )


def test_leave_one_subject_out_scores_each_participant_and_exports_rows():
    result = leave_one_subject_out(_feature_sets(), fit_model=_fit_nearest_mean, include_class_scores=True)

    assert len(result.folds) == 3
    assert result.mean_accuracy == 1.0
    assert result.mean_balanced_accuracy == 1.0
    assert all(fold.chance_accuracy == 0.5 for fold in result.folds)

    fold_rows = result.fold_rows()
    assert {row["test_participant"] for row in fold_rows} == {"s0", "s1", "s2"}
    assert all(row["top2_accuracy"] == 1.0 for row in fold_rows)

    prediction_rows = result.prediction_rows()
    assert len(prediction_rows) == 12
    assert prediction_rows[0]["sample_id"] == 10
    assert prediction_rows[0]["correct"] is True

    summary = summarize_cross_subject_folds(result)
    assert summary["n_outer_folds"] == 3
    assert summary["participants_above_chance"] == 3


def test_nested_leave_one_subject_out_selects_best_candidate_by_inner_loso():
    candidates = (
        DecoderCandidate("constant-zero", _fit_constant_zero),
        DecoderCandidate("nearest-mean", _fit_nearest_mean),
    )

    result = nested_leave_one_subject_out(_feature_sets(), candidates=candidates)

    assert result.mean_balanced_accuracy == 1.0
    assert [outer.selected_candidate_index for outer in result.outer_folds] == [1, 1, 1]
    assert [outer.selected_candidate_name for outer in result.outer_folds] == ["nearest-mean", "nearest-mean", "nearest-mean"]

    selection_rows = result.selection_rows()
    assert len(selection_rows) == 6
    assert sum(row["selected"] for row in selection_rows) == 3


def test_rank_true_labels_returns_nan_for_unscoreable_labels():
    ranks = rank_true_labels(
        true_labels=np.array([2, 3, 4]),
        class_scores=np.array(
            [
                [0.2, 0.5, 0.1],
                [0.1, 0.2, 0.2],
                [0.8, 0.1, 0.0],
            ]
        ),
        score_classes=np.array([1, 2, 3]),
    )

    assert ranks[0] == 1.0
    assert ranks[1] == 1.0
    assert np.isnan(ranks[2])
