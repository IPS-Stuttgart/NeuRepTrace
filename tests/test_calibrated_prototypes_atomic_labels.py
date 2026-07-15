from __future__ import annotations

import numpy as np

from neureptrace.decoding.calibrated_prototypes import fit_calibrated_prototype_decoder


def test_calibrated_prototype_supports_missing_class_labels() -> None:
    source_features = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [3.0, 3.0],
            [3.1, 3.0],
        ],
        dtype=float,
    )
    source_labels = np.empty(4, dtype=object)
    source_labels[:] = [float("nan"), float("nan"), "right", "right"]
    calibration_features = np.asarray([[0.05, 0.0], [3.05, 3.0]], dtype=float)
    calibration_labels = np.empty(2, dtype=object)
    calibration_labels[:] = [float("nan"), "right"]
    eval_features = np.asarray([[0.02, 0.0], [3.08, 3.0]], dtype=float)

    result = fit_calibrated_prototype_decoder(
        source_features=source_features,
        source_labels=source_labels,
        calibration_features=calibration_features,
        calibration_labels=calibration_labels,
        eval_features=eval_features,
    )

    assert result.classes.shape == (2,)
    assert np.isnan(result.classes[0])
    assert result.classes[1] == "right"
    assert result.calibration_counts.tolist() == [1, 1]
    assert np.isnan(result.predictions[0])
    assert result.predictions[1] == "right"
    assert np.all(np.isfinite(result.prototypes))


def test_calibrated_prototype_preserves_composite_labels_as_atomic_values() -> None:
    source_features = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [4.0, 4.0],
            [4.1, 4.0],
        ],
        dtype=float,
    )
    source_labels = [
        ["face", "left"],
        ["face", "left"],
        ["object", "right"],
        ["object", "right"],
    ]
    calibration_features = np.asarray([[0.05, 0.0], [4.05, 4.0]], dtype=float)
    calibration_labels = [["face", "left"], ["object", "right"]]
    eval_features = np.asarray([[0.02, 0.0], [4.08, 4.0]], dtype=float)

    result = fit_calibrated_prototype_decoder(
        source_features=source_features,
        source_labels=source_labels,
        calibration_features=calibration_features,
        calibration_labels=calibration_labels,
        eval_features=eval_features,
    )

    assert result.classes.shape == (2,)
    assert result.classes.tolist() == [("face", "left"), ("object", "right")]
    assert result.predictions.tolist() == [("face", "left"), ("object", "right")]
    assert result.metadata["calibrated_prototype_n_classes"] == 2
