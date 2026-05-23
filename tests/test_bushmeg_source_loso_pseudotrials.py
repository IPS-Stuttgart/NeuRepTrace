from __future__ import annotations

import numpy as np

from neureptrace.bushmeg_source_loso import _candidate_grid, _source_pseudotrial_training_features


def test_source_pseudotrials_average_subject_class_cells_and_preserve_weights():
    features = np.asarray(
        [[0.0], [2.0], [10.0], [14.0], [100.0], [102.0], [110.0], [114.0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=int)
    subject_ids = np.asarray(["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"], dtype=object)
    weights = np.ones(features.shape[0], dtype=float)

    pseudo_features, pseudo_labels, pseudo_weights = _source_pseudotrial_training_features(
        features,
        labels,
        subject_ids,
        pseudotrials_per_subject_class=1,
        sample_weight=weights,
    )

    expected_features = np.asarray([[1.0], [12.0], [101.0], [112.0]], dtype=np.float32)
    expected_labels = np.asarray([0, 1, 0, 1], dtype=int)
    np.testing.assert_allclose(pseudo_features, expected_features)
    np.testing.assert_array_equal(pseudo_labels, expected_labels)
    assert pseudo_weights is not None
    np.testing.assert_allclose(pseudo_weights, np.ones(4))


def test_candidate_grid_expands_source_pseudotrial_values():
    config = {
        "preprocessing": {"window_size": 0.100},
        "decoding": {
            "classifier": "logistic",
            "emission_mode": "uncalibrated",
            "feature_preprocessor": "none",
            "pca_components": None,
            "tuning_c_grid": "1.0",
        },
        "source_loso": {
            "candidate_grid": {
                "pseudotrials_per_subject_class": [0, 2],
                "decoders": ["logistic"],
                "emission_modes": ["uncalibrated"],
                "feature_preprocessors": ["none"],
                "pca_components": [None],
                "temporal_bins": [1],
                "feature_kinds": ["evoked"],
                "c_grid": [1.0],
                "window_sets": [{"name": "single", "centers": [0.175], "window_size": 0.100}],
            }
        },
    }

    candidates = _candidate_grid(config)

    assert {candidate.pseudotrials_per_subject_class for candidate in candidates} == {0, 2}
    assert any("__pt2__" in candidate.name for candidate in candidates)
