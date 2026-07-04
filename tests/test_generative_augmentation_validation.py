from __future__ import annotations

import numpy as np
import pytest

import neureptrace  # noqa: F401 - installs runtime validation patches
from neureptrace.decoding.generative_augmentation import augment_training_features, generative_augmentation_config


def test_generative_augmentation_rejects_nan_train_features() -> None:
    config = generative_augmentation_config(method="source_gaussian", synthetic_per_class=1, random_state=13)
    train_features = np.asarray([[0.0, 1.0], [np.nan, 2.0]])
    train_labels = np.asarray([0, 1])

    with pytest.raises(ValueError, match="train_features must contain only finite values"):
        augment_training_features(train_features, train_labels, config=config)


def test_generative_augmentation_rejects_inf_target_features() -> None:
    config = generative_augmentation_config(method="target_style_gaussian", synthetic_per_class=1, random_state=13)
    train_features = np.asarray([[0.0, 1.0], [1.0, 2.0]])
    train_labels = np.asarray([0, 1])
    target_features = np.asarray([[0.5, np.inf], [1.5, 2.5]])

    with pytest.raises(ValueError, match="target_features must contain only finite values"):
        augment_training_features(train_features, train_labels, config=config, target_features=target_features)


def test_generative_augmentation_rejects_nan_target_calibration_features() -> None:
    config = generative_augmentation_config(method="target_calibrated_gaussian", synthetic_per_class=1, random_state=13)
    train_features = np.asarray([[0.0, 1.0], [1.0, 2.0]])
    train_labels = np.asarray([0, 1])
    target_calibration_features = np.asarray([[0.5, 1.5], [1.5, np.nan]])
    target_calibration_labels = np.asarray([0, 1])

    with pytest.raises(ValueError, match="target_calibration_features must contain only finite values"):
        augment_training_features(
            train_features,
            train_labels,
            config=config,
            target_calibration_features=target_calibration_features,
            target_calibration_labels=target_calibration_labels,
        )
