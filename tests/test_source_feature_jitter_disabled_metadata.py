from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_jitter import augment_source_with_feature_jitter


def test_disabled_jitter_metadata_reports_actual_output_rows_when_originals_are_preserved() -> None:
    features = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    labels = np.asarray([0, 1])

    result = augment_source_with_feature_jitter(features, labels, config={"synthetic_per_class": 0, "preserve_original": False})

    assert result.features.shape == features.shape
    assert result.metadata["source_feature_jitter"] is False
    assert result.metadata["source_feature_jitter_n_synthetic_rows"] == 0
    assert result.metadata["source_feature_jitter_n_output_rows"] == features.shape[0]


def test_zero_noise_disabled_jitter_metadata_reports_actual_output_rows() -> None:
    features = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
    labels = np.asarray([0, 0, 1])

    result = augment_source_with_feature_jitter(
        features,
        labels,
        config={"synthetic_per_class": 2, "noise_scale": 0.0, "preserve_original": False},
    )

    assert result.features.shape == features.shape
    assert result.n_synthetic == 0
    assert result.metadata["source_feature_jitter_n_output_rows"] == features.shape[0]
