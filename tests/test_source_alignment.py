import numpy as np
import pytest

from neureptrace.decoding.source_alignment import align_train_test_features, source_alignment_config


def _rotated_subject_features(seed=13):
    rng = np.random.default_rng(seed)
    labels_one_subject = np.repeat(np.arange(3), 8)
    prototypes = np.array(
        [
            [2.0, 0.0, 0.0, 0.5],
            [0.0, 2.0, 0.5, 0.0],
            [0.5, 0.0, 2.0, 0.0],
        ]
    )
    features = []
    labels = []
    subjects = []
    for subject in range(3):
        q, _r = np.linalg.qr(rng.normal(size=(4, 4)))
        subject_features = prototypes[labels_one_subject] @ q + 0.02 * rng.normal(size=(labels_one_subject.size, 4))
        features.append(subject_features)
        labels.append(labels_one_subject)
        subjects.extend([f"s{subject}"] * labels_one_subject.size)
    return np.vstack(features), np.concatenate(labels), np.asarray(subjects, dtype=object)


def _mean_subject_class_distance(features, labels, subjects):
    means = []
    for subject in np.unique(subjects):
        subject_means = []
        for label in np.unique(labels):
            mask = (subjects == subject) & (labels == label)
            subject_means.append(features[mask].mean(axis=0))
        means.append(np.vstack(subject_means))
    distances = []
    for left in range(len(means)):
        for right in range(left + 1, len(means)):
            distances.append(float(np.linalg.norm(means[left] - means[right])))
    return float(np.mean(distances))


def test_none_alignment_reproduces_raw_features():
    train_features = np.array([[1.0, 0.0], [0.0, 1.0]])
    test_features = np.array([[0.5, 0.5]])

    result = align_train_test_features(
        train_features=train_features,
        train_labels=np.array([0, 1]),
        train_subject_ids=np.array(["a", "b"]),
        test_features=test_features,
        config=source_alignment_config(method="none"),
    )

    np.testing.assert_allclose(result.train_features, train_features)
    np.testing.assert_allclose(result.test_features, test_features)
    assert result.metadata["alignment_method"] == "none"


def test_strict_alignment_rejects_target_labels():
    with pytest.raises(ValueError, match="target labels"):
        align_train_test_features(
            train_features=np.ones((4, 2)),
            train_labels=np.array([0, 1, 0, 1]),
            train_subject_ids=np.array(["a", "a", "b", "b"]),
            test_features=np.ones((2, 2)),
            target_labels=np.array([0, 1]),
            config=source_alignment_config(method="procrustes"),
        )


@pytest.mark.parametrize("method", ["procrustes", "hyperalignment", "mcca"])
def test_source_alignment_methods_expose_group_projection_metadata(method):
    train_features, train_labels, train_subjects = _rotated_subject_features()
    raw_distance = _mean_subject_class_distance(train_features, train_labels, train_subjects)

    result = align_train_test_features(
        train_features=train_features,
        train_labels=train_labels,
        train_subject_ids=train_subjects,
        test_features=train_features[:6],
        config=source_alignment_config(method=method, components=2),
    )

    aligned_distance = _mean_subject_class_distance(result.train_features, train_labels, train_subjects)
    assert result.train_features.shape == (train_features.shape[0], 2)
    assert result.test_features.shape == (6, 2)
    assert result.metadata["alignment_method"] == method
    assert result.metadata["alignment_target_projection"] == "group_projection"
    assert result.metadata["alignment_n_components"] == 2
    assert aligned_distance < raw_distance


def test_class_repetition_cap_is_capped_by_available_counts():
    train_features, train_labels, train_subjects = _rotated_subject_features()

    result = align_train_test_features(
        train_features=train_features,
        train_labels=train_labels,
        train_subject_ids=train_subjects,
        test_features=train_features[:2],
        config=source_alignment_config(
            method="mcca",
            anchor_mode="class_repetition",
            repetition_cap=99,
            components=4,
        ),
    )

    assert result.metadata["alignment_anchor_mode"] == "class_repetition"
    assert result.metadata["alignment_repetitions_per_class"] == 8
