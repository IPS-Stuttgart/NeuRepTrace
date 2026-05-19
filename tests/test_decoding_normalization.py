import numpy as np
import pytest

from neureptrace.decoding.normalization import (
    baseline_channel_whitening_matrix_from_features,
    baseline_whiten_features,
    baseline_zscore_features,
    normalize_normalization,
    normalize_subject_features,
    subject_zscore_features,
    trial_zscore_features,
)


def test_normalize_normalization_accepts_hyphen_alias():
    assert normalize_normalization("subject-baseline-z") == "subject_baseline_z"


@pytest.mark.parametrize("mode", ["unknown", "subject-bad"])
def test_normalize_normalization_rejects_unknown_modes(mode):
    with pytest.raises(ValueError, match="normalization"):
        normalize_normalization(mode)


def test_subject_zscore_features_handles_constant_columns():
    features = np.asarray([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])

    normalized = subject_zscore_features(features)

    np.testing.assert_allclose(normalized[:, 0], 0.0)
    np.testing.assert_allclose(np.mean(normalized[:, 1]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.std(normalized[:, 1]), 1.0, atol=1e-12)
    assert not np.shares_memory(normalized, features)


def test_subject_zscore_features_can_use_reference_distribution():
    features = np.asarray([[3.0, 6.0]])
    reference = np.asarray([[1.0, 2.0], [5.0, 10.0]])

    normalized = subject_zscore_features(features, reference_features=reference)

    np.testing.assert_allclose(normalized, [[0.0, 0.0]])


def test_trial_zscore_features_normalizes_rows():
    features = np.asarray([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])

    normalized = trial_zscore_features(features)

    np.testing.assert_allclose(normalized[0], 0.0)
    np.testing.assert_allclose(np.mean(normalized[1]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.std(normalized[1]), 1.0, atol=1e-12)


def test_baseline_zscore_features_uses_baseline_statistics():
    features = np.asarray([[3.0, 5.0], [5.0, 9.0]])

    normalized = baseline_zscore_features(features, baseline_feature_mean=[[1.0, 1.0]], baseline_feature_std=[[2.0, 4.0]])

    np.testing.assert_allclose(normalized, [[1.0, 1.0], [2.0, 2.0]])


def test_normalize_subject_features_dispatches_baseline_z():
    normalized = normalize_subject_features(
        [[3.0, 5.0]],
        "subject-baseline-z",
        baseline_feature_mean=[[1.0, 1.0]],
        baseline_feature_std=[[2.0, 4.0]],
    )

    np.testing.assert_allclose(normalized, [[1.0, 1.0]])


def test_baseline_whiten_features_handles_sensor_mean_mode():
    normalized = baseline_whiten_features(
        [[2.0, 3.0]],
        baseline_feature_mean=[[1.0, 1.0]],
        baseline_whitening_matrix=[[2.0, 0.0], [0.0, 3.0]],
        feature_mode="sensor_mean",
    )

    np.testing.assert_allclose(normalized, [[2.0, 6.0]])


def test_baseline_whiten_features_applies_matrix_to_each_feature_block():
    normalized = baseline_whiten_features(
        [[2.0, 3.0, 10.0, 20.0]],
        baseline_feature_mean=[[0.0, 0.0, 0.0, 0.0]],
        baseline_whitening_matrix=[[2.0, 0.0], [0.0, 3.0]],
        feature_mode="sensor_flat",
    )

    np.testing.assert_allclose(normalized, [[4.0, 9.0, 20.0, 60.0]])


def test_baseline_channel_whitening_matrix_from_features_is_finite_for_degenerate_baseline():
    whitening = baseline_channel_whitening_matrix_from_features([[1.0, 2.0], [1.0, 2.0]])

    assert whitening.shape == (2, 2)
    assert np.all(np.isfinite(whitening))
    np.testing.assert_allclose(whitening, whitening.T)
