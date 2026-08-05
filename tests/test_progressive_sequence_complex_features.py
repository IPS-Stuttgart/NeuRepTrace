from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.progressive_sequence_finetune import (
    NestedTrialCalibrationSplit,
    TorchProgressiveSequenceClassifier,
    fit_progressive_sequence_target_calibrated_decoder,
    pack_complete_trial_events,
)


def test_pack_complete_trial_events_rejects_complex_features() -> None:
    features = np.asarray(
        [
            [1.0 + 2.0j, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="real-valued features"):
        pack_complete_trial_events(
            features,
            trial_ids=["trial-1"] * 4,
            press_positions=[1, 2, 3, 4],
        )


def test_calibrated_decoder_rejects_object_complex_features_before_training() -> None:
    source_features = np.asarray([[[1.0 + 1.0j], [2.0], [3.0], [4.0]]], dtype=object)
    source_labels = np.asarray([[0, 1, 2, 3]])
    target_features = np.zeros((2, 4, 1), dtype=float)
    target_labels = np.asarray([[0, 1, 2, 3], [0, 1, 2, 3]])
    split = NestedTrialCalibrationSplit(
        calibration_indices=np.asarray([0]),
        evaluation_indices=np.asarray([1]),
        calibration_pool_indices=np.asarray([0]),
        per_stratum=1,
        max_per_stratum=1,
        seed=13,
    )

    with pytest.raises(ValueError, match="source_features must contain real-valued features"):
        fit_progressive_sequence_target_calibrated_decoder(
            source_features=source_features,
            source_labels=source_labels,
            target_features=target_features,
            target_labels=target_labels,
            split=split,
        )


def test_classifier_rejects_complex_source_features_before_torch_import() -> None:
    classifier = TorchProgressiveSequenceClassifier()
    features = np.asarray([[[1.0 + 1.0j], [2.0], [3.0], [4.0]]])
    labels = np.asarray([[0, 1, 2, 3]])

    with pytest.raises(ValueError, match="source_features must contain real-valued features"):
        classifier.fit_source(features, labels)
