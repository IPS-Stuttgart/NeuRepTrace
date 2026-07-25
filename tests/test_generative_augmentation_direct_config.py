from __future__ import annotations

import numpy as np

from neureptrace.decoding.generative_augmentation import GenerativeAugmentationConfig, augment_training_features


def _features_and_labels() -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray([[0.0], [1.0], [10.0], [11.0]], dtype=float)
    labels = np.asarray([0, 0, 1, 1])
    return features, labels


def test_direct_generative_config_normalizes_disabled_method_alias() -> None:
    features, labels = _features_and_labels()

    result = augment_training_features(
        features,
        labels,
        config=GenerativeAugmentationConfig(method="off", synthetic_per_class=2),
    )

    np.testing.assert_array_equal(result.features, features)
    np.testing.assert_array_equal(result.labels, labels)
    assert result.n_synthetic == 0
    assert result.metadata["generative_augmentation_method"] == "none"
    assert result.metadata["generative_augmentation_enabled"] is False


def test_direct_generative_config_normalizes_aliases_and_numeric_strings() -> None:
    features, labels = _features_and_labels()

    result = augment_training_features(
        features,
        labels,
        config=GenerativeAugmentationConfig(
            method="gaussian",
            synthetic_per_class="1",  # type: ignore[arg-type]
            random_state="7",  # type: ignore[arg-type]
        ),
    )

    assert result.n_synthetic == 2
    assert result.metadata["generative_augmentation_method"] == "source_gaussian"
    assert result.metadata["generative_augmentation_protocol_category"] == 1
    assert result.metadata["generative_augmentation_protocol_note"] == "source-only synthetic feature augmentation"
