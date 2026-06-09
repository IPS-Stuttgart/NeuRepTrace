import numpy as np
import pytest

from neureptrace.decoding.source_alignment import (
    ORACLE_TARGET_CALIBRATED_ALIGNMENT,
    align_train_test_features,
    normalize_source_alignment_method,
    source_alignment_config,
)


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


def _mean_source_target_class_distance(source_features, source_labels, target_features, target_labels):
    distances = []
    for label in np.unique(source_labels):
        source_mean = source_features[source_labels == label].mean(axis=0)
        target_mean = target_features[target_labels == label].mean(axis=0)
        distances.append(float(np.linalg.norm(source_mean - target_mean)))
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


def test_strict_alignment_rejects_target_anchor_values():
    with pytest.raises(ValueError, match="target anchor values"):
        align_train_test_features(
            train_features=np.ones((4, 2)),
            train_labels=np.array([0, 1, 0, 1]),
            train_subject_ids=np.array(["a", "a", "b", "b"]),
            test_features=np.ones((2, 2)),
            train_anchor_values=np.array(["stim-a", "stim-b", "stim-a", "stim-b"]),
            target_anchor_values=np.array(["stim-a", "stim-b"]),
            config=source_alignment_config(method="procrustes", anchor_mode="stimulus_id_mean"),
        )


@pytest.mark.parametrize(
    ("alias", "method"),
    [
        ("euclidean_alignment", "euclidean"),
        ("coral_alignment", "coral"),
        ("target_covariance_alignment", "target_baseline_covariance"),
        ("sensor_covariance_normalization", "subject_sensor_covariance"),
    ],
)
def test_unsupervised_alignment_method_aliases(alias, method):
    assert normalize_source_alignment_method(alias) == method


def test_alignment_times_same_decode_window_metadata():
    config = source_alignment_config(method="procrustes", times="same_decode_window")
    metadata = config.static_metadata()

    assert config.same_decode_window is True
    assert config.times == ()
    assert metadata["alignment_times"] == "same_decode_window"
    assert metadata["alignment_window_mode"] == "same_decode_window"
    assert metadata["alignment_same_decode_window"] is True


def test_oracle_alignment_requires_target_labels():
    train_features, train_labels, train_subjects = _rotated_subject_features()
    with pytest.raises(ValueError, match="requires held-out target labels"):
        align_train_test_features(
            train_features=train_features,
            train_labels=train_labels,
            train_subject_ids=train_subjects,
            test_features=train_features[:6],
            config=source_alignment_config(
                method="procrustes",
                target_projection=ORACLE_TARGET_CALIBRATED_ALIGNMENT,
            ),
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
    assert result.diagnostics["alignment_method"] == method
    assert result.diagnostics["sample_mode"] == "class_mean"
    assert result.diagnostics["n_source_subjects"] == 3
    assert result.diagnostics["n_classes"] == 3
    assert result.diagnostics["n_alignment_rows"] == 3
    assert result.diagnostics["requested_components"] == 2
    assert result.diagnostics["actual_components"] == 2
    assert result.diagnostics["feature_dim"] == train_features.shape[1]
    assert result.diagnostics["decode_feature_dim"] == 2
    assert result.diagnostics["uses_channel_projection_collapse"] is True
    assert np.isfinite(result.diagnostics["anchor_row_correlation_before"])
    assert np.isfinite(result.diagnostics["anchor_row_correlation_after"])
    assert np.isfinite(result.diagnostics["source_inner_decoding_before_alignment"])
    assert np.isfinite(result.diagnostics["source_inner_decoding_after_alignment"])
    assert result.diagnostics["source_inner_raw_balanced_accuracy"] == result.diagnostics["source_inner_decoding_before_alignment"]
    assert result.diagnostics["source_inner_aligned_balanced_accuracy"] == result.diagnostics["source_inner_decoding_after_alignment"]
    assert np.isfinite(result.diagnostics["source_inner_aligned_minus_raw"])
    assert result.diagnostics["source_inner_validation_type"] == "strict_source_loso_nearest_centroid_group_projection"
    assert result.diagnostics["target_transform_type"] == "source_group_projection"
    assert aligned_distance < raw_distance


@pytest.mark.parametrize(
    ("method", "target_transform_type"),
    [
        ("euclidean", "unlabeled_target_covariance_whitening"),
        ("coral", "unlabeled_target_covariance_recoloring"),
        ("target_baseline_covariance", "unlabeled_target_pooled_source_to_target_covariance"),
        ("subject_sensor_covariance", "unlabeled_target_covariance_whitening"),
    ],
)
def test_unsupervised_covariance_alignment_uses_no_class_anchors(method, target_transform_type):
    train_features, train_labels, train_subjects = _rotated_subject_features(seed=29)
    target_features = train_features[train_subjects == "s2"] * np.array([1.5, 0.5, 1.2, 0.8])

    result = align_train_test_features(
        train_features=train_features[train_subjects != "s2"],
        train_labels=train_labels[train_subjects != "s2"],
        train_subject_ids=train_subjects[train_subjects != "s2"],
        test_features=target_features,
        config=source_alignment_config(method=method),
    )

    assert result.train_features.shape[1] == train_features.shape[1]
    assert result.test_features.shape == target_features.shape
    assert result.metadata["alignment_anchor_value_source"] == "unlabeled_covariance"
    assert result.metadata["alignment_uses_unlabeled_target_data"] is True
    assert result.metadata["alignment_uses_class_labels"] is False
    assert result.metadata["alignment_target_labels_used"] is False
    assert result.diagnostics["sample_mode"] == "unlabeled_covariance"
    assert result.diagnostics["uses_unlabeled_target_data"] is True
    assert result.diagnostics["target_transform_type"] == target_transform_type
    assert result.diagnostics["covariance_alignment_estimator"] == "full"
    assert np.isfinite(result.diagnostics["source_inner_raw_balanced_accuracy"])


def test_unsupervised_covariance_alignment_rejects_oracle_target_projection():
    with pytest.raises(ValueError, match="does not support oracle target labels"):
        source_alignment_config(method="coral", target_projection=ORACLE_TARGET_CALIBRATED_ALIGNMENT)


@pytest.mark.parametrize("method", ["procrustes", "hyperalignment", "mcca"])
def test_stimulus_anchor_values_are_distinct_from_decoder_labels(method):
    train_features, train_labels, train_subjects = _rotated_subject_features(seed=11)
    repetitions = np.tile(np.tile(np.arange(8), 3), 3)
    train_anchors = np.asarray([f"stim-{label}-{rep % 2}" for label, rep in zip(train_labels, repetitions, strict=True)])

    result = align_train_test_features(
        train_features=train_features,
        train_labels=train_labels,
        train_subject_ids=train_subjects,
        train_anchor_values=train_anchors,
        test_features=train_features[:6],
        config=source_alignment_config(
            method=method,
            anchor_mode="stimulus_id_mean",
            anchor_column="stim_file",
            components=2,
        ),
    )

    assert result.train_features.shape == (train_features.shape[0], 2)
    assert result.metadata["alignment_anchor_mode"] == "stimulus_id_mean"
    assert result.metadata["alignment_anchor_column"] == "stim_file"
    assert result.metadata["alignment_anchor_value_source"] == "metadata"
    assert result.metadata["alignment_common_anchor_count"] == 6
    assert result.metadata["alignment_anchor_rows_dropped"] == 0
    assert result.metadata["alignment_target_anchor_values_used"] is False


@pytest.mark.parametrize("method", ["procrustes", "hyperalignment", "mcca"])
def test_oracle_target_calibrated_alignment_is_debug_upper_bound(method):
    features, labels, subjects = _rotated_subject_features(seed=41)
    source_mask = subjects != "s2"
    target_mask = subjects == "s2"
    config_kwargs = {"method": method, "components": 2}

    strict = align_train_test_features(
        train_features=features[source_mask],
        train_labels=labels[source_mask],
        train_subject_ids=subjects[source_mask],
        test_features=features[target_mask],
        config=source_alignment_config(**config_kwargs),
    )
    oracle = align_train_test_features(
        train_features=features[source_mask],
        train_labels=labels[source_mask],
        train_subject_ids=subjects[source_mask],
        test_features=features[target_mask],
        target_labels=labels[target_mask],
        config=source_alignment_config(
            **config_kwargs,
            target_projection=ORACLE_TARGET_CALIBRATED_ALIGNMENT,
        ),
    )

    strict_distance = _mean_source_target_class_distance(
        strict.train_features,
        labels[source_mask],
        strict.test_features,
        labels[target_mask],
    )
    oracle_distance = _mean_source_target_class_distance(
        oracle.train_features,
        labels[source_mask],
        oracle.test_features,
        labels[target_mask],
    )
    assert oracle_distance < strict_distance
    assert oracle.metadata["alignment_target_projection"] == ORACLE_TARGET_CALIBRATED_ALIGNMENT
    assert oracle.metadata["alignment_oracle_target_calibrated"] is True
    assert oracle.metadata["alignment_debug_upper_bound"] is True
    assert oracle.metadata["alignment_valid_for_benchmark"] is False
    assert oracle.metadata["alignment_target_labels_used"] is True
    assert oracle.metadata["alignment_protocol_note"] == "debug upper bound only; not valid for benchmark"


def test_oracle_target_calibrated_alignment_accepts_target_anchor_values():
    features, labels, subjects = _rotated_subject_features(seed=43)
    repetitions = np.tile(np.tile(np.arange(8), 3), 3)
    anchors = np.asarray([f"stim-{label}-{rep % 2}" for label, rep in zip(labels, repetitions, strict=True)])
    source_mask = subjects != "s2"
    target_mask = subjects == "s2"

    oracle = align_train_test_features(
        train_features=features[source_mask],
        train_labels=labels[source_mask],
        train_subject_ids=subjects[source_mask],
        train_anchor_values=anchors[source_mask],
        test_features=features[target_mask],
        target_anchor_values=anchors[target_mask],
        config=source_alignment_config(
            method="procrustes",
            anchor_mode="stimulus_id_mean",
            anchor_column="stim_file",
            components=2,
            target_projection=ORACLE_TARGET_CALIBRATED_ALIGNMENT,
        ),
    )

    assert oracle.metadata["alignment_target_projection"] == ORACLE_TARGET_CALIBRATED_ALIGNMENT
    assert oracle.metadata["alignment_target_labels_used"] is False
    assert oracle.metadata["alignment_target_anchor_values_used"] is True
    assert oracle.metadata["alignment_valid_for_benchmark"] is False


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
