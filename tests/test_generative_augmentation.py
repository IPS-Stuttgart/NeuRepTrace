import numpy as np
import pytest

from neureptrace.decoding.generative_augmentation import (
    TARGET_CALIBRATED_GENERATIVE_PROTOCOL,
    UNLABELED_TARGET_GENERATIVE_PROTOCOL,
    augment_training_features,
    generative_augmentation_config,
    make_generative_augmented_fit_model,
)


class _RecordingModel:
    def __init__(self, features, labels):
        self.n_rows = features.shape[0]
        self.labels = np.asarray(labels)


def _toy_features():
    features = np.array(
        [
            [-2.0, -1.0],
            [-1.5, -1.0],
            [-1.0, -0.5],
            [1.0, 0.5],
            [1.5, 1.0],
            [2.0, 1.0],
        ]
    )
    labels = np.array([0, 0, 0, 1, 1, 1])
    return features, labels


def test_source_gaussian_appends_deterministic_source_only_samples():
    features, labels = _toy_features()
    config = generative_augmentation_config(method="source_gaussian", synthetic_per_class=2, random_state=7)

    first = augment_training_features(features, labels, config=config)
    second = augment_training_features(features, labels, config=config)

    assert first.features.shape == (10, 2)
    assert first.labels.tolist().count(0) == 5
    assert first.labels.tolist().count(1) == 5
    assert first.n_synthetic == 4
    assert first.synthetic_mask.tolist() == [False] * 6 + [True] * 4
    assert first.metadata["generative_augmentation_protocol_category"] == 1
    assert first.metadata["generative_augmentation_valid_for_strict_source_only"] is True
    np.testing.assert_allclose(first.features, second.features)


def test_target_style_gaussian_uses_unlabeled_target_features_only():
    features, labels = _toy_features()
    target_features = features + np.array([10.0, -3.0])
    config = generative_augmentation_config(method="target_style_gaussian", synthetic_per_class=1, random_state=3)

    augmented = augment_training_features(features, labels, config=config, target_features=target_features)

    assert augmented.features.shape == (8, 2)
    assert augmented.metadata["generative_augmentation_protocol"] == UNLABELED_TARGET_GENERATIVE_PROTOCOL
    assert augmented.metadata["generative_augmentation_protocol_category"] == 2
    assert augmented.metadata["generative_augmentation_uses_unlabeled_target_data"] is True
    assert augmented.metadata["generative_augmentation_uses_target_labels"] is False


def test_target_style_gaussian_requires_target_features():
    features, labels = _toy_features()
    config = generative_augmentation_config(method="target_style_gaussian", synthetic_per_class=1)

    with pytest.raises(ValueError, match="requires unlabeled target_features"):
        augment_training_features(features, labels, config=config)


def test_scored_target_labels_are_rejected():
    features, labels = _toy_features()
    config = generative_augmentation_config(method="source_gaussian", synthetic_per_class=1)

    with pytest.raises(ValueError, match="never accepts scored target_labels"):
        augment_training_features(features, labels, config=config, target_labels=labels)


def test_target_calibrated_gaussian_requires_disjoint_calibration_labels():
    features, labels = _toy_features()
    config = generative_augmentation_config(method="target_calibrated_gaussian", synthetic_per_class=1)

    with pytest.raises(ValueError, match="requires disjoint target_calibration_features"):
        augment_training_features(features, labels, config=config)


def test_target_calibrated_gaussian_records_category_3_metadata():
    features, labels = _toy_features()
    config = generative_augmentation_config(
        method="target_calibrated_gaussian",
        synthetic_per_class=1,
        target_calibration_weight=1.0,
        random_state=5,
    )
    calibration_features = np.array([[-4.0, -4.0], [4.0, 4.0]])
    calibration_labels = np.array([0, 1])

    augmented = augment_training_features(
        features,
        labels,
        config=config,
        target_calibration_features=calibration_features,
        target_calibration_labels=calibration_labels,
    )

    assert augmented.features.shape == (8, 2)
    assert augmented.metadata["generative_augmentation_protocol"] == TARGET_CALIBRATED_GENERATIVE_PROTOCOL
    assert augmented.metadata["generative_augmentation_protocol_category"] == 3
    assert augmented.metadata["generative_augmentation_uses_target_labels"] is True
    assert augmented.metadata["generative_augmentation_valid_for_strict_source_only"] is False


def test_fit_model_wrapper_augments_current_training_rows_and_attaches_metadata():
    features, labels = _toy_features()
    config = generative_augmentation_config(method="source_gaussian", synthetic_per_class=1, random_state=11)
    fit_model = make_generative_augmented_fit_model(lambda x, y: _RecordingModel(x, y), config=config)

    model = fit_model(features, labels)

    assert model.n_rows == 8
    assert model.labels.tolist().count(0) == 4
    assert model.labels.tolist().count(1) == 4
    assert model.generative_augmentation_metadata_["generative_augmentation_n_synthetic"] == 2
