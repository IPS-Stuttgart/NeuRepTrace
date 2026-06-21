import numpy as np
import pytest

import neureptrace  # noqa: F401  # ensure runtime alignment patches are installed
from neureptrace.decoding.source_alignment import (
    PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT,
    align_train_test_features,
    source_alignment_config,
)


def _rotated_subject_features(seed=97):
    rng = np.random.default_rng(seed)
    labels_one_subject = np.repeat(np.arange(3), 4)
    prototypes = np.array(
        [
            [2.0, 0.0, 0.0, 0.5],
            [0.0, 2.0, 0.5, 0.0],
            [0.5, 0.0, 2.0, 0.0],
        ]
    )
    features = []
    labels = []
    subjects = []
    for subject in range(3):
        q, _r = np.linalg.qr(rng.normal(size=(4, 4)))
        subject_features = prototypes[labels_one_subject] @ q + 0.02 * rng.normal(
            size=(labels_one_subject.size, 4)
        )
        features.append(subject_features)
        labels.append(labels_one_subject)
        subjects.extend([f"s{subject}"] * labels_one_subject.size)
    return np.vstack(features), np.concatenate(labels), np.asarray(subjects, dtype=object)


@pytest.mark.parametrize("method", ["procrustes", "hyperalignment", "mcca"])
def test_pseudo_label_target_calibrated_class_repetition_uses_calibration_cap(method):
    features, labels, subjects = _rotated_subject_features()
    source_mask = subjects != "s2"
    target_positions = np.flatnonzero(subjects == "s2")
    classes = np.unique(labels)
    calibration_positions = np.asarray(
        [target_positions[labels[target_positions] == class_label][0] for class_label in classes]
    )
    config = source_alignment_config(
        method=method,
        anchor_mode="class_repetition",
        repetition_cap=4,
        components=2,
        target_projection=PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT,
        target_calibration_per_anchor=1,
    )

    static_metadata = config.static_metadata()
    assert static_metadata["alignment_pseudo_label_target_calibrated"] is True
    assert static_metadata["alignment_strict_source_only"] is False
    assert static_metadata["alignment_valid_for_strict_source_only"] is False
    assert static_metadata["alignment_valid_for_benchmark"] is False
    assert static_metadata["alignment_uses_unlabeled_target_data"] is True
    assert "not valid for strict benchmark comparisons" in static_metadata["alignment_protocol_note"]

    result = align_train_test_features(
        train_features=features[source_mask],
        train_labels=labels[source_mask],
        train_subject_ids=subjects[source_mask],
        test_features=features[target_positions],
        target_calibration_features=features[calibration_positions],
        target_calibration_labels=labels[calibration_positions],
        config=config,
    )

    assert result.metadata["alignment_target_projection"] == PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT
    assert result.metadata["alignment_pseudo_label_target_calibrated"] is True
    assert result.metadata["alignment_valid_for_benchmark"] is False
    assert result.metadata["alignment_valid_for_strict_source_only"] is False
    assert result.metadata["alignment_repetitions_per_class"] == 1
    assert result.metadata["alignment_target_alignment_rows"] == 3
    assert result.metadata["alignment_target_pseudo_labels_used"] is True
    assert result.metadata["alignment_target_anchor_values_used"] is True
    assert result.test_features.shape == (target_positions.size, 2)
