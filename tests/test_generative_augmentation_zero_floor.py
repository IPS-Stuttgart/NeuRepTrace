from __future__ import annotations

import numpy as np

from neureptrace.decoding.generative_augmentation import augment_training_features, generative_augmentation_config


def test_target_style_zero_covariance_floor_handles_singular_source_covariance() -> None:
    source_features = np.zeros((4, 2), dtype=float)
    source_labels = np.asarray([0, 0, 1, 1])
    target_features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ]
    )
    config = generative_augmentation_config(
        method="target_style_gaussian",
        synthetic_per_class=1,
        covariance_floor=0.0,
        random_state=7,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        augmented = augment_training_features(
            source_features,
            source_labels,
            config=config,
            target_features=target_features,
        )

    synthetic = augmented.features[augmented.synthetic_mask]
    expected = np.repeat(np.mean(target_features, axis=0, keepdims=True), 2, axis=0)
    assert synthetic.shape == (2, 2)
    assert np.isfinite(synthetic).all()
    np.testing.assert_allclose(synthetic, expected)
