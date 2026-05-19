import numpy as np
import pytest
from sklearn.neighbors import NearestCentroid

from neureptrace.decoding.shared_space import (
    SharedSpaceConfig,
    SubjectFeatureSet,
    evaluate_loso_shared_space,
    fit_loso_shared_space_fold,
    make_loso_shared_space_folds,
)


def _synthetic_subjects(*, with_alignment=False):
    rng = np.random.default_rng(42)
    labels = np.tile(np.arange(3), 8)
    latent = np.eye(3)[labels] + 0.03 * rng.normal(size=(labels.size, 3))
    subjects = []
    for subject in range(4):
        mixing = rng.normal(size=(3, 9))
        features = latent @ mixing + 0.03 * rng.normal(size=(labels.size, 9))
        if with_alignment:
            alignment_latent = np.eye(3)[labels] + 0.03 * rng.normal(size=(labels.size, 3))
            alignment_features = alignment_latent @ mixing + 0.03 * rng.normal(size=(labels.size, 9))
            subjects.append(SubjectFeatureSet(subject, features, labels, alignment_features=alignment_features, alignment_labels=labels))
        else:
            subjects.append(SubjectFeatureSet(subject, features, labels))
    return subjects


def test_loso_hyperalignment_folds_return_projected_train_and_test_matrices():
    config = SharedSpaceConfig(method="hyperalignment", sample_mode="class_repetition", n_repetitions_per_class=4, n_components=3, hyperalignment_iterations=2)

    folds = make_loso_shared_space_folds(_synthetic_subjects(), config=config)

    assert len(folds) == 4
    assert {fold.test_subject for fold in folds} == {0, 1, 2, 3}
    for fold in folds:
        assert fold.method == "hyperalignment"
        assert fold.target_transform == "target_unsupervised"
        assert fold.train_features.shape == (72, fold.n_components)
        assert fold.test_features.shape == (24, fold.n_components)
        assert fold.train_labels.shape == (72,)
        assert fold.test_labels.shape == (24,)
        assert fold.alignment_classes.tolist() == [0, 1, 2]


def test_evaluate_loso_shared_space_scores_arbitrary_estimator():
    config = SharedSpaceConfig(method="hyperalignment", sample_mode="class_repetition", n_repetitions_per_class=4, n_components=3, hyperalignment_iterations=2)

    scores = evaluate_loso_shared_space(
        _synthetic_subjects(),
        config=config,
        fit_estimator=lambda x, y: NearestCentroid().fit(x, y),
    )

    assert len(scores) == 4
    for score in scores:
        assert score.n_train_trials == 72
        assert score.n_test_trials == 24
        assert score.predicted_labels.shape == score.true_labels.shape
        assert 0.0 <= score.accuracy <= 1.0
        assert 0.0 <= score.balanced_accuracy <= 1.0


def test_mcca_target_labeled_uses_explicit_target_alignment_data():
    config = SharedSpaceConfig(
        method="mcca",
        target_transform="target_labeled",
        sample_mode="class_repetition",
        n_repetitions_per_class=4,
        n_components=3,
        mcca_regularization=1e-5,
    )

    fold = fit_loso_shared_space_fold(_synthetic_subjects(with_alignment=True), test_subject=0, config=config)

    assert fold.method == "mcca"
    assert fold.target_transform == "target_labeled"
    assert fold.n_target_alignment_rows == 12
    assert fold.train_features.shape == (72, fold.n_components)
    assert fold.test_features.shape == (24, fold.n_components)


def test_target_labeled_rejects_implicit_use_of_scored_test_labels():
    config = SharedSpaceConfig(method="mcca", target_transform="target_labeled", n_components=2)

    with pytest.raises(ValueError, match="target_labeled requires alignment_features"):
        fit_loso_shared_space_fold(_synthetic_subjects(with_alignment=False), test_subject=0, config=config)


def test_label_shuffle_changes_training_labels_without_touching_test_labels():
    config = SharedSpaceConfig(method="hyperalignment", sample_mode="class_repetition", n_repetitions_per_class=4, n_components=3, hyperalignment_iterations=2)
    subjects = _synthetic_subjects()

    baseline = fit_loso_shared_space_fold(subjects, test_subject=0, config=config)
    shuffled = fit_loso_shared_space_fold(subjects, test_subject=0, config=config, label_shuffle_seed=7)

    assert not np.array_equal(baseline.train_labels, shuffled.train_labels)
    assert np.array_equal(baseline.test_labels, shuffled.test_labels)
