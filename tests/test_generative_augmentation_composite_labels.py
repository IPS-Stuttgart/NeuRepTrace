import numpy as np

from neureptrace.decoding.generative_augmentation import augment_training_features, generative_augmentation_config


def test_source_gaussian_preserves_composite_tuple_labels():
    features = np.array(
        [
            [-2.0, -1.0],
            [-1.5, -0.5],
            [1.5, 0.5],
            [2.0, 1.0],
        ]
    )
    labels = [("subject-a", "left"), ("subject-a", "left"), ("subject-a", "right"), ("subject-a", "right")]
    config = generative_augmentation_config(method="source_gaussian", synthetic_per_class=1, random_state=5)

    augmented = augment_training_features(features, labels, config=config)

    assert augmented.features.shape == (6, 2)
    assert augmented.synthetic_mask.tolist() == [False, False, False, False, True, True]
    assert augmented.labels[:4].tolist() == labels
    assert augmented.labels.tolist().count(("subject-a", "left")) == 3
    assert augmented.labels.tolist().count(("subject-a", "right")) == 3


def test_target_calibrated_gaussian_accepts_composite_tuple_labels():
    features = np.array(
        [
            [-2.0, -1.0],
            [-1.5, -0.5],
            [1.5, 0.5],
            [2.0, 1.0],
        ]
    )
    labels = [("subject-a", "left"), ("subject-a", "left"), ("subject-a", "right"), ("subject-a", "right")]
    calibration_features = np.array([[-3.0, -2.0], [3.0, 2.0]])
    calibration_labels = [("subject-a", "left"), ("subject-a", "right")]
    config = generative_augmentation_config(method="target_calibrated_gaussian", synthetic_per_class=1, random_state=7)

    augmented = augment_training_features(
        features,
        labels,
        config=config,
        target_calibration_features=calibration_features,
        target_calibration_labels=calibration_labels,
    )

    assert augmented.features.shape == (6, 2)
    assert augmented.labels.tolist().count(("subject-a", "left")) == 3
    assert augmented.labels.tolist().count(("subject-a", "right")) == 3
    assert augmented.metadata["generative_augmentation_protocol_category"] == 3
