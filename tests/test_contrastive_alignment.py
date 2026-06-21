import numpy as np
import pytest

from neureptrace.decoding.contrastive_alignment import fit_contrastive_alignment
from neureptrace.decoding.source_alignment import (
    GROUP_PROJECTION_TARGET_CENTERED,
    TARGET_CALIBRATED_ALIGNMENT,
    align_train_test_features,
    normalize_source_alignment_method,
    source_alignment_anchor_availability,
    source_alignment_config,
)


def _rotated_subject_features(seed=13, n_subjects=4):
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
    for subject in range(n_subjects):
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


def test_contrastive_method_aliases_are_registered():
    assert normalize_source_alignment_method("contrastive") == "contrastive"
    assert normalize_source_alignment_method("contrastive-subject-alignment") == "contrastive"
    assert normalize_source_alignment_method("supervised_contrastive") == "contrastive"


def test_core_contrastive_alignment_collapses_source_anchors():
    features, labels, subjects = _rotated_subject_features(seed=19, n_subjects=3)
    features_by_subject = {subject: features[subjects == subject] for subject in np.unique(subjects)}
    labels_by_subject = {subject: labels[subjects == subject] for subject in np.unique(subjects)}

    model, alignment = fit_contrastive_alignment(
        features_by_subject,
        labels_by_subject,
        sample_mode="class_mean",
        n_components=2,
    )
    transformed = np.vstack([model.transform(subject, features_by_subject[subject]) for subject in model.subject_ids])

    assert model.n_components == 2
    assert alignment.n_alignment_rows == 3
    assert transformed.shape == (features.shape[0], 2)
    assert _mean_subject_class_distance(transformed, labels, subjects) < _mean_subject_class_distance(features, labels, subjects)


def test_source_alignment_contrastive_group_projection_metadata():
    features, labels, subjects = _rotated_subject_features(seed=23, n_subjects=3)
    raw_distance = _mean_subject_class_distance(features, labels, subjects)

    result = align_train_test_features(
        train_features=features,
        train_labels=labels,
        train_subject_ids=subjects,
        test_features=features[:6],
        config=source_alignment_config(method="contrastive", components=2),
    )

    aligned_distance = _mean_subject_class_distance(result.train_features, labels, subjects)
    assert result.train_features.shape == (features.shape[0], 2)
    assert result.test_features.shape == (6, 2)
    assert result.metadata["alignment_method"] == "contrastive"
    assert result.metadata["alignment_target_projection"] == "group_projection"
    assert result.metadata["alignment_strict_source_only"] is True
    assert result.metadata["alignment_uses_class_labels"] is True
    assert result.metadata["alignment_target_labels_used"] is False
    assert result.metadata["alignment_valid_for_strict_source_only"] is True
    assert result.metadata["alignment_n_components"] == 2
    assert result.diagnostics["alignment_method"] == "contrastive"
    assert result.diagnostics["sample_mode"] == "class_mean"
    assert result.diagnostics["target_transform_type"] == "source_group_contrastive_projection"
    assert result.diagnostics["source_inner_validation_type"] == "strict_source_loso_nearest_centroid_group_projection"
    assert np.isfinite(result.diagnostics["source_inner_raw_balanced_accuracy"])
    assert np.isfinite(result.diagnostics["source_inner_aligned_balanced_accuracy"])
    assert aligned_distance < raw_distance


def test_contrastive_target_centered_group_projection_uses_unlabeled_target_mean():
    features, labels, subjects = _rotated_subject_features(seed=29, n_subjects=3)
    source_mask = subjects != "s2"
    target_mask = subjects == "s2"
    shifted_target = features[target_mask] + np.array([25.0, -10.0, 5.0, 3.0])
    base_kwargs = {
        "train_features": features[source_mask],
        "train_labels": labels[source_mask],
        "train_subject_ids": subjects[source_mask],
        "test_features": shifted_target,
    }

    strict = align_train_test_features(
        **base_kwargs,
        config=source_alignment_config(method="contrastive", components=2),
        compute_source_inner_diagnostics=False,
    )
    centered = align_train_test_features(
        **base_kwargs,
        config=source_alignment_config(
            method="contrastive",
            components=2,
            target_projection=GROUP_PROJECTION_TARGET_CENTERED,
        ),
        compute_source_inner_diagnostics=False,
    )

    assert centered.metadata["alignment_target_projection"] == GROUP_PROJECTION_TARGET_CENTERED
    assert centered.metadata["alignment_strict_source_only"] is False
    assert centered.metadata["alignment_uses_unlabeled_target_data"] is True
    assert centered.metadata["alignment_valid_for_strict_source_only"] is False
    assert centered.diagnostics["uses_unlabeled_target_data"] is True
    assert centered.diagnostics["target_transform_type"] == "source_group_contrastive_projection_target_centered"
    assert not np.allclose(strict.test_features, centered.test_features)


def test_contrastive_target_calibrated_projection_uses_disjoint_target_labels():
    features, labels, subjects = _rotated_subject_features(seed=31, n_subjects=4)
    source_mask = subjects != "s3"
    target_mask = subjects == "s3"
    target_features = features[target_mask]
    target_labels = labels[target_mask]
    calibration_mask = np.zeros(target_labels.shape[0], dtype=bool)
    for label in np.unique(target_labels):
        calibration_mask[np.flatnonzero(target_labels == label)[0]] = True

    result = align_train_test_features(
        train_features=features[source_mask],
        train_labels=labels[source_mask],
        train_subject_ids=subjects[source_mask],
        test_features=target_features[~calibration_mask],
        target_calibration_features=target_features[calibration_mask],
        target_calibration_labels=target_labels[calibration_mask],
        config=source_alignment_config(
            method="contrastive",
            components=2,
            target_projection=TARGET_CALIBRATED_ALIGNMENT,
        ),
        compute_source_inner_diagnostics=False,
    )

    assert result.test_features.shape == (target_features[~calibration_mask].shape[0], 2)
    assert result.metadata["alignment_target_calibrated"] is True
    assert result.metadata["alignment_target_labels_used"] is True
    assert result.metadata["alignment_target_alignment_rows"] == 3
    assert result.metadata["alignment_valid_for_strict_source_only"] is False
    assert result.diagnostics["target_transform_type"] == "target_calibrated_contrastive_ridge_projection"


def test_contrastive_anchor_availability_reports_source_common_anchors():
    row = source_alignment_anchor_availability(
        train_labels=np.array([0, 1, 0, 1, 0, 1]),
        train_subject_ids=np.array(["s0", "s0", "s1", "s1", "s2", "s2"], dtype=object),
        config=source_alignment_config(method="contrastive", components=1),
    )

    assert row["prefit_status"] == "ok"
    assert row["alignment_method"] == "contrastive"
    assert row["n_common_source_anchors"] == 2
    assert row["estimated_alignment_rows"] == 2


def test_contrastive_rejects_target_labels_outside_oracle_mode():
    features, labels, subjects = _rotated_subject_features(seed=37, n_subjects=3)

    with pytest.raises(ValueError, match="target labels"):
        align_train_test_features(
            train_features=features,
            train_labels=labels,
            train_subject_ids=subjects,
            test_features=features[:6],
            target_labels=labels[:6],
            config=source_alignment_config(method="contrastive", components=2),
        )
