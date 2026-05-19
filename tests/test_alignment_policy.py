from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.alignment_policy import AlignmentPlan, fit_alignment_policy, normalize_alignment_plan


def _synthetic_labeled_subjects(seed=13):
    rng = np.random.default_rng(seed)
    n_classes = 6
    n_repetitions = 4
    labels = np.repeat(np.arange(n_classes), n_repetitions)
    class_latent = rng.normal(size=(n_classes, 3))
    latent = class_latent[labels] + 0.05 * rng.normal(size=(labels.shape[0], 3))

    features = {}
    labels_by_subject = {}
    for subject in range(4):
        mixing = rng.normal(size=(3, 8))
        features[subject] = latent @ mixing + 0.03 * rng.normal(size=(labels.shape[0], 8))
        labels_by_subject[subject] = labels.copy()
    return features, labels_by_subject


def _source_target_split():
    features, labels = _synthetic_labeled_subjects()
    source_features = {subject: matrix for subject, matrix in features.items() if subject != 3}
    source_labels = {subject: vector for subject, vector in labels.items() if subject != 3}
    return source_features, source_labels, features[3], labels[3]


def test_none_alignment_policy_returns_identity_rows() -> None:
    policy = fit_alignment_policy(
        {"subject": np.array([[1.0, 2.0], [3.0, 4.0]])},
        {"subject": np.array([0, 1])},
    )

    features = np.array([[5.0, 6.0]])
    np.testing.assert_allclose(policy.transform_source("subject", features), features)
    np.testing.assert_allclose(policy.transform_target(features), features)
    assert policy.provenance["alignment_method"] == "none"
    assert policy.provenance["alignment_target_transform"] == "none"


def test_normalize_alignment_plan_supports_pymegdec_procrustes_alias() -> None:
    plan = normalize_alignment_plan(AlignmentPlan(method="train-class-procrustes", target_policy="group"))

    assert plan.method == "hyperalignment"
    assert plan.target_policy == "group"
    assert plan.fit_data == "source_only"


def test_hyperalignment_group_policy_uses_group_projection() -> None:
    source_features, source_labels, target_features, _target_labels = _source_target_split()
    plan = AlignmentPlan(method="hyperalignment", target_policy="group", n_components=3, n_iterations=3)

    policy = fit_alignment_policy(source_features, source_labels, plan=plan, target_id=3)

    transformed = policy.transform_target(target_features)
    expected = policy.model.transform_group(target_features)
    np.testing.assert_allclose(transformed, expected)
    assert transformed.shape == (target_features.shape[0], 3)
    assert policy.provenance["alignment_fit_data"] == "source_only"
    assert policy.provenance["alignment_target_centering"] == "training_group_mean"


def test_mcca_unsupervised_centering_uses_supplied_target_mean() -> None:
    source_features, source_labels, target_features, _target_labels = _source_target_split()
    plan = AlignmentPlan(method="mcca", target_policy="unsupervised_centering", n_components=3, regularization=1e-5)

    with pytest.raises(ValueError, match="target_centering_features"):
        fit_alignment_policy(source_features, source_labels, plan=plan)

    policy = fit_alignment_policy(
        source_features,
        source_labels,
        plan=plan,
        target_centering_features=target_features,
        target_id=3,
    )

    expected = policy.model.transform_group(target_features, feature_mean=np.mean(target_features, axis=0))
    np.testing.assert_allclose(policy.transform_target(target_features), expected)
    assert policy.provenance["alignment_target_transform"] == "group_projection"
    assert policy.provenance["alignment_target_centering"] == "target_unsupervised_mean"


def test_mcca_calibration_policy_fits_target_projection_without_adding_target_to_source_model() -> None:
    source_features, source_labels, target_features, target_labels = _source_target_split()
    plan = AlignmentPlan(method="mcca", target_policy="calibration", n_components=3, regularization=1e-5)

    policy = fit_alignment_policy(
        source_features,
        source_labels,
        plan=plan,
        target_features=target_features,
        target_labels=target_labels,
        target_id=3,
    )

    source_transformed = policy.transform_source(0, source_features[0])
    target_transformed = policy.transform_target(target_features)

    assert policy.model.subject_ids == (0, 1, 2)
    assert policy.target_projection is not None
    assert source_transformed.shape == (target_features.shape[0], 3)
    assert target_transformed.shape == (target_features.shape[0], 3)
    assert policy.provenance["alignment_target_transform"] == "target_calibration_projection"
    assert policy.provenance["alignment_target_calibration_rows"] == 6


def test_label_shuffle_policies_are_deterministic_and_recorded() -> None:
    source_features, source_labels, target_features, target_labels = _source_target_split()
    plan = AlignmentPlan(
        method="mcca",
        target_policy="calibration",
        source_label_policy="shuffle",
        target_label_policy="shuffle",
        random_state=7,
        n_components=2,
        regularization=1e-5,
    )

    first = fit_alignment_policy(source_features, source_labels, plan=plan, target_features=target_features, target_labels=target_labels)
    second = fit_alignment_policy(source_features, source_labels, plan=plan, target_features=target_features, target_labels=target_labels)

    np.testing.assert_allclose(first.transform_target(target_features), second.transform_target(target_features))
    assert first.provenance["alignment_source_label_policy"] == "shuffle"
    assert first.provenance["alignment_target_label_policy"] == "shuffle"


def test_target_calibration_policy_requires_target_labels() -> None:
    source_features, source_labels, target_features, _target_labels = _source_target_split()
    plan = AlignmentPlan(method="mcca", target_policy="calibration")

    with pytest.raises(ValueError, match="target_features and target_labels"):
        fit_alignment_policy(source_features, source_labels, plan=plan, target_features=target_features)
