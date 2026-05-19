import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from neureptrace.decoding.cross_subject import (
    CrossSubjectCandidate,
    SubjectFeatureSet,
    leave_one_subject_out_decoding,
    nested_leave_one_subject_out_decoding,
)


class NearestCentroid:
    def fit(self, features, labels):
        self.classes_ = np.unique(labels)
        self.centroids_ = np.vstack([np.mean(features[labels == label], axis=0) for label in self.classes_])
        return self

    def decision_function(self, features):
        features = np.asarray(features, dtype=float)
        return -np.linalg.norm(features[:, None, :] - self.centroids_[None, :, :], axis=2)

    def predict(self, features):
        return self.classes_[np.argmax(self.decision_function(features), axis=1)]


def _centroid_fit(features, labels):
    return NearestCentroid().fit(features, labels)


def _dummy_fit(features, labels):
    return DummyClassifier(strategy="most_frequent").fit(features, labels)


def _feature_sets(n_subjects=4):
    base_features = np.array(
        [
            [-2.0, -0.2],
            [-2.2, 0.1],
            [-1.8, 0.2],
            [2.0, -0.1],
            [2.1, 0.2],
            [1.9, 0.0],
        ]
    )
    labels = np.array(["face", "face", "face", "tool", "tool", "tool"])
    return tuple(
        SubjectFeatureSet(
            subject=f"s{subject}",
            features=base_features + 0.01 * subject,
            labels=labels,
            trial_ids=np.arange(10 * subject, 10 * subject + labels.shape[0]),
        )
        for subject in range(n_subjects)
    )


def test_leave_one_subject_out_decoding_scores_all_held_out_subjects():
    result = leave_one_subject_out_decoding(_feature_sets(), fit_model=_centroid_fit)

    assert len(result.outer) == 4
    assert len(result.predictions) == 24
    assert all(row["accuracy"] == 1.0 for row in result.outer)
    assert result.group_summary[0]["balanced_accuracy_mean"] == 1.0
    assert {row["candidate_name"] for row in result.outer} == {"candidate"}


def test_nested_leave_one_subject_out_decoding_selects_best_candidate():
    candidates = (
        CrossSubjectCandidate("dummy", _dummy_fit),
        CrossSubjectCandidate("centroid", _centroid_fit, metadata={"classifier": "nearest-centroid"}),
    )

    result = nested_leave_one_subject_out_decoding(_feature_sets(), candidates=candidates)

    assert len(result.inner_validation) == 4 * 2 * 3
    assert len(result.outer) == 4
    assert all(row["selected_candidate_index"] == 2 for row in result.selected)
    assert all(row["candidate_name"] == "centroid" for row in result.outer)
    assert all(row["selected_classifier"] == "nearest-centroid" for row in result.outer)
    assert result.group_summary[0]["balanced_accuracy_mean"] == 1.0


def test_nested_leave_one_subject_out_decoding_validates_subject_count():
    with pytest.raises(ValueError, match="At least three subjects"):
        nested_leave_one_subject_out_decoding(
            _feature_sets(n_subjects=2),
            candidates=(CrossSubjectCandidate("centroid", _centroid_fit),),
        )
