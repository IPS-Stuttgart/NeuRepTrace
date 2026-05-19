import numpy as np
import pytest

from neureptrace.decoding import (
    baseline_whiten_features,
    covariance_matrix,
    fit_feature_normalizer,
    normalize_feature_normalization,
    shrink_covariance,
    trial_zscore_features,
    whitening_matrix_from_baseline_covariance,
)


def test_subject_z_normalizer_fits_feature_statistics_without_mutating_input():
    features = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    original = features.copy()

    normalizer = fit_feature_normalizer(features, mode="subject_z")
    transformed = normalizer.transform(features)

    np.testing.assert_allclose(features, original)
    np.testing.assert_allclose(np.mean(transformed, axis=0), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.std(transformed, axis=0), [1.0, 1.0])


def test_trial_zscore_features_is_row_local_and_handles_constant_rows():
    transformed = trial_zscore_features(np.array([[1.0, 2.0, 3.0], [7.0, 7.0, 7.0]]))

    np.testing.assert_allclose(transformed[0], [-1.22474487139, 0.0, 1.22474487139])
    np.testing.assert_allclose(transformed[1], [0.0, 0.0, 0.0])


def test_baseline_z_tiles_channel_statistics_for_flat_feature_blocks():
    features = np.array([[10.0, 20.0, 12.0, 22.0]])
    baseline_features = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])

    normalizer = fit_feature_normalizer(features, mode="subject_baseline_z", baseline_features=baseline_features)
    transformed = normalizer.transform(features)

    mean = np.tile(np.mean(baseline_features, axis=0, keepdims=True), 2)
    scale = np.tile(np.std(baseline_features, axis=0, keepdims=True), 2)
    np.testing.assert_allclose(transformed, (features - mean) / scale)


def test_baseline_whiten_applies_provided_matrix_per_feature_block():
    features = np.array([[3.0, 5.0, 4.0, 6.0]])
    normalizer = fit_feature_normalizer(
        features,
        mode="subject_baseline_whiten",
        baseline_mean=np.array([1.0, 2.0]),
        whitening_matrix=np.diag([2.0, 3.0]),
    )

    transformed = normalizer.transform(features)

    np.testing.assert_allclose(transformed, [[4.0, 9.0, 6.0, 12.0]])


def test_baseline_whiten_features_validates_block_width():
    with pytest.raises(ValueError, match="integer multiple"):
        baseline_whiten_features(np.ones((2, 5)), np.eye(2))


def test_covariance_shrinkage_and_whitening_are_symmetric():
    covariance = covariance_matrix(np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]]))
    shrunk = shrink_covariance(covariance, shrinkage=0.25)
    whitening = whitening_matrix_from_baseline_covariance(covariance, shrinkage=0.25)

    np.testing.assert_allclose(covariance, covariance.T)
    np.testing.assert_allclose(shrunk, shrunk.T)
    np.testing.assert_allclose(whitening, whitening.T)


def test_normalization_validators_reject_unsupported_modes_and_missing_baseline():
    assert normalize_feature_normalization("subject-baseline-z") == "subject_baseline_z"
    with pytest.raises(ValueError, match="normalization"):
        normalize_feature_normalization("global_z")
    with pytest.raises(ValueError, match="baseline"):
        fit_feature_normalizer(np.ones((2, 3)), mode="subject_baseline_z")
