import numpy as np
import pytest

from neureptrace.decoding.source_alignment import (
    _transform_unsupervised_covariance_alignment_by_subject,
    align_train_test_features,
    normalize_source_alignment_method,
    source_alignment_anchor_availability,
    source_alignment_config,
)


def _shifted_domains():
    rng = np.random.default_rng(17)
    labels = np.tile(np.array([0, 1]), 12)
    base = np.column_stack([labels.astype(float) * 2.0, 1.0 - labels.astype(float)])
    source_a = base + np.array([-4.0, 3.0]) + 0.03 * rng.normal(size=base.shape)
    source_b = base + np.array([-3.5, 2.7]) + 0.03 * rng.normal(size=base.shape)
    target = base + np.array([4.0, -2.0]) + 0.03 * rng.normal(size=base.shape)
    train_features = np.vstack([source_a, source_b])
    train_labels = np.concatenate([labels, labels])
    subjects = np.array(["s0"] * labels.size + ["s1"] * labels.size, dtype=object)
    return train_features, train_labels, subjects, target


@pytest.mark.parametrize(
    "alias",
    [
        "optimal_transport",
        "wasserstein",
        "sinkhorn",
        "sinkhorn_ot",
        "barycentric_transport",
    ],
)
def test_optimal_transport_alignment_aliases(alias):
    assert normalize_source_alignment_method(alias) == "sinkhorn_transport"


def test_sinkhorn_transport_alignment_is_unlabeled_category2():
    config = source_alignment_config(method="optimal_transport")
    metadata = config.static_metadata()

    assert config.method == "sinkhorn_transport"
    assert metadata["alignment_strict_source_only"] is False
    assert metadata["alignment_uses_unlabeled_target_data"] is True
    assert metadata["alignment_uses_class_labels"] is False
    assert metadata["alignment_valid_for_benchmark"] is False
    assert metadata["alignment_valid_for_strict_source_only"] is False
    assert metadata["alignment_protocol"] == "unlabeled_target_optimal_transport_alignment"
    assert "Sinkhorn barycentric optimal transport" in metadata["alignment_protocol_note"]


def test_sinkhorn_transport_moves_source_distribution_toward_unlabeled_target():
    train_features, train_labels, subjects, target_features = _shifted_domains()
    raw_distance = float(np.linalg.norm(train_features.mean(axis=0) - target_features.mean(axis=0)))

    result = align_train_test_features(
        train_features=train_features,
        train_labels=train_labels,
        train_subject_ids=subjects,
        test_features=target_features,
        config=source_alignment_config(method="wasserstein"),
        compute_source_inner_diagnostics=False,
    )

    aligned_distance = float(np.linalg.norm(result.train_features.mean(axis=0) - target_features.mean(axis=0)))
    assert aligned_distance < raw_distance * 0.5
    np.testing.assert_allclose(result.test_features, target_features)
    assert result.metadata["alignment_method"] == "sinkhorn_transport"
    assert result.metadata["alignment_anchor_value_source"] == "unlabeled_target_distribution"
    assert result.metadata["alignment_target_labels_used"] is False
    assert result.metadata["alignment_target_anchor_values_used"] is False
    assert result.metadata["target_transform_type"] == "unlabeled_target_sinkhorn_barycentric_transport"
    assert result.diagnostics["sample_mode"] == "unlabeled_optimal_transport"
    assert result.diagnostics["uses_unlabeled_target_data"] is True
    assert result.diagnostics["covariance_alignment_estimator"] == "sinkhorn_barycentric"


def test_sinkhorn_transport_helper_preserves_subject_keys_and_target_rows():
    train_features, _train_labels, subjects, target_features = _shifted_domains()
    features_by_subject = {subject: train_features[subjects == subject] for subject in tuple(dict.fromkeys(subjects.tolist()))}

    transformed_by_subject, transformed_target, metadata = _transform_unsupervised_covariance_alignment_by_subject(
        features_by_subject,
        target_features,
        method="sinkhorn_transport",
    )

    assert set(transformed_by_subject) == set(features_by_subject)
    for subject, transformed in transformed_by_subject.items():
        assert transformed.shape == features_by_subject[subject].shape
    np.testing.assert_allclose(transformed_target, target_features)
    assert metadata["alignment_uses_unlabeled_target_data"] is True
    assert metadata["alignment_covariance_method"] == "sinkhorn_transport"
    assert metadata["covariance_alignment_estimator"] == "sinkhorn_barycentric"


def test_sinkhorn_anchor_availability_reports_no_anchor_requirement():
    row = source_alignment_anchor_availability(
        train_labels=np.array([0, 1, 0, 1]),
        train_subject_ids=np.array(["s0", "s0", "s1", "s1"]),
        config=source_alignment_config(method="sinkhorn"),
    )

    assert row["prefit_status"] == "no_anchors_required"
    assert row["sample_mode"] == "unlabeled_optimal_transport"
    assert row["source_anchor_value_source"] == "unlabeled_target_distribution"
    assert row["alignment_protocol"] == "unlabeled_target_optimal_transport_alignment"
