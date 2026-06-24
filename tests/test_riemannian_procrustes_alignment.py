import numpy as np

import neureptrace.decoding.source_alignment as source_alignment
from neureptrace.decoding.source_alignment import (
    RIEMANNIAN_PROCRUSTES_METHOD,
    _transform_unsupervised_covariance_alignment_by_subject,
    align_train_test_features,
    normalize_source_alignment_method,
    source_alignment_anchor_availability,
    source_alignment_config,
)


def _orthogonal(seed: int, n_features: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(n_features, n_features)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def _rpa_demo_features():
    rng = np.random.default_rng(7)
    base = rng.normal(size=(48, 4))
    features_by_subject = {
        "s0": 0.6 * base @ _orthogonal(1) + np.array([4.0, -2.0, 0.5, 1.0]),
        "s1": 1.7 * base @ _orthogonal(2) + np.array([-3.0, 1.0, 2.0, -1.0]),
    }
    target = 1.2 * base @ _orthogonal(3) + np.array([0.5, 2.5, -1.0, 3.0])
    return features_by_subject, target


def _dispersion(features: np.ndarray) -> float:
    centered = features - np.mean(features, axis=0)
    covariance = centered.T @ centered / max(1, features.shape[0] - 1)
    return float(np.sqrt(np.trace(covariance) / features.shape[1]))


def test_source_alignment_package_all_keeps_legacy_public_exports():
    exports = set(source_alignment.__all__)

    assert "align_train_test_features" in exports
    assert "source_alignment_config" in exports
    assert "SOURCE_ALIGNMENT_METHODS" in exports
    assert "RIEMANNIAN_PROCRUSTES_METHOD" in exports


def test_riemannian_procrustes_aliases_are_category2_methods():
    for alias in (
        "rpa",
        "full-rpa",
        "riemannian-procrustes",
        "riemannian_procrustes_analysis",
        "procrustes-covariance",
    ):
        assert normalize_source_alignment_method(alias) == RIEMANNIAN_PROCRUSTES_METHOD

    config = source_alignment_config(method="rpa")
    metadata = config.static_metadata()

    assert config.method == RIEMANNIAN_PROCRUSTES_METHOD
    assert metadata["alignment_method"] == RIEMANNIAN_PROCRUSTES_METHOD
    assert metadata["alignment_protocol"] == "unlabeled_target_covariance_alignment"
    assert metadata["alignment_uses_unlabeled_target_data"] is True
    assert metadata["alignment_uses_class_labels"] is False
    assert metadata["alignment_valid_for_benchmark"] is False
    assert metadata["alignment_valid_for_strict_source_only"] is False


def test_riemannian_procrustes_uses_unlabeled_target_distribution_without_labels():
    features_by_subject, target = _rpa_demo_features()

    transformed_by_subject, transformed_target, metadata = _transform_unsupervised_covariance_alignment_by_subject(
        features_by_subject,
        target,
        method="riemannian-procrustes",
    )

    np.testing.assert_allclose(transformed_target, target)
    assert metadata["alignment_uses_unlabeled_target_data"] is True
    assert metadata["alignment_covariance_method"] == RIEMANNIAN_PROCRUSTES_METHOD
    assert metadata["target_transform_type"] == "unlabeled_target_riemannian_procrustes_recenter_scale_rotate"
    assert metadata["riemannian_procrustes_rotation_used"] is True
    assert metadata["covariance_alignment_estimator"] == "full"

    target_mean = np.mean(target, axis=0)
    target_dispersion = _dispersion(target)
    for transformed in transformed_by_subject.values():
        np.testing.assert_allclose(np.mean(transformed, axis=0), target_mean, atol=1e-10)
        np.testing.assert_allclose(_dispersion(transformed), target_dispersion, rtol=1e-10, atol=1e-10)


def test_align_train_test_features_reports_rpa_as_unlabeled_target_adaptive():
    features_by_subject, target = _rpa_demo_features()
    train_features = np.vstack([features_by_subject["s0"], features_by_subject["s1"]])
    train_subjects = np.asarray(["s0"] * 48 + ["s1"] * 48, dtype=object)
    train_labels = np.tile(np.repeat(np.arange(3), 16), 2)

    result = align_train_test_features(
        train_features=train_features,
        train_labels=train_labels,
        train_subject_ids=train_subjects,
        test_features=target,
        config=source_alignment_config(method="riemannian_procrustes"),
        compute_source_inner_diagnostics=False,
    )

    assert result.train_features.shape == train_features.shape
    assert result.test_features.shape == target.shape
    assert result.metadata["alignment_method"] == RIEMANNIAN_PROCRUSTES_METHOD
    assert result.metadata["alignment_anchor_value_source"] == "unlabeled_covariance"
    assert result.metadata["alignment_target_labels_used"] is False
    assert result.metadata["alignment_target_anchor_values_used"] is False
    assert result.metadata["alignment_uses_unlabeled_target_data"] is True
    assert result.metadata["alignment_valid_for_strict_source_only"] is False
    assert result.diagnostics["sample_mode"] == "unlabeled_covariance"
    assert result.diagnostics["uses_unlabeled_target_data"] is True
    assert result.diagnostics["target_transform_type"] == "unlabeled_target_riemannian_procrustes_recenter_scale_rotate"


def test_anchor_availability_treats_rpa_as_anchor_free_unlabeled_alignment():
    row = source_alignment_anchor_availability(
        train_labels=np.array([0, 1, 0, 1]),
        train_subject_ids=np.array(["s0", "s0", "s1", "s1"], dtype=object),
        config=source_alignment_config(method="rpa"),
    )

    assert row["alignment_method"] == RIEMANNIAN_PROCRUSTES_METHOD
    assert row["source_anchor_value_source"] == "unlabeled_covariance"
    assert row["prefit_status"] == "no_anchors_required"
