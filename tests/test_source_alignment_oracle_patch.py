import numpy as np
import pytest

import neureptrace  # noqa: F401  # ensure runtime alignment patches are installed
from neureptrace.decoding.source_alignment import (
    ORACLE_TARGET_CALIBRATED_ALIGNMENT,
    align_train_test_features,
    source_alignment_config,
)


def _rotated_subject_features(seed=109, n_subjects=3):
    rng = np.random.default_rng(seed)
    labels_one_subject = np.repeat(np.arange(3), 5)
    prototypes = np.array(
        [
            [2.0, 0.0, 0.0, 0.5],
            [0.0, 2.0, 0.5, 0.0],
            [0.5, 0.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    features = []
    labels = []
    subjects = []
    for subject in range(n_subjects):
        q, _r = np.linalg.qr(rng.normal(size=(4, 4)))
        subject_features = prototypes[labels_one_subject] @ q + 0.02 * rng.normal(
            size=(labels_one_subject.size, 4)
        )
        features.append(subject_features)
        labels.append(labels_one_subject)
        subjects.extend([f"s{subject}"] * labels_one_subject.size)
    return np.vstack(features), np.concatenate(labels), np.asarray(subjects, dtype=object)


@pytest.mark.parametrize(
    ("method", "expected_transform"),
    [
        ("procrustes", "oracle_target_calibrated_template_procrustes"),
        ("hyperalignment", "oracle_target_calibrated_template_procrustes"),
        ("mcca", "oracle_target_calibrated_template_ridge_least_squares"),
    ],
)
def test_oracle_alignment_explicitly_marks_target_projection(method, expected_transform):
    features, labels, subjects = _rotated_subject_features()
    source_mask = subjects != "s2"
    target_mask = subjects == "s2"
    config = source_alignment_config(
        method=method,
        components=2,
        target_projection=ORACLE_TARGET_CALIBRATED_ALIGNMENT,
    )

    static_metadata = config.static_metadata()
    assert static_metadata["alignment_protocol"] == ORACLE_TARGET_CALIBRATED_ALIGNMENT
    assert static_metadata["alignment_debug_upper_bound"] is True
    assert static_metadata["alignment_valid_for_benchmark"] is False
    assert static_metadata["alignment_valid_for_strict_source_only"] is False
    assert "scored held-out target" in static_metadata["alignment_protocol_note"]

    result = align_train_test_features(
        train_features=features[source_mask],
        train_labels=labels[source_mask],
        train_subject_ids=subjects[source_mask],
        test_features=features[target_mask],
        target_labels=labels[target_mask],
        config=config,
    )

    assert result.metadata["alignment_target_projection"] == ORACLE_TARGET_CALIBRATED_ALIGNMENT
    assert result.metadata["alignment_target_projection_fit"] == expected_transform
    assert result.metadata["alignment_target_labels_used"] is True
    assert result.metadata["alignment_target_anchor_values_used"] is True
    assert result.metadata["alignment_oracle_target_calibrated"] is True
    assert result.metadata["alignment_debug_upper_bound"] is True
    assert result.metadata["alignment_valid_for_benchmark"] is False
    assert result.metadata["alignment_valid_for_strict_source_only"] is False
    assert result.diagnostics["target_transform_type"] == expected_transform
    assert result.diagnostics["source_inner_validation_type"] == "source_loso_nearest_centroid_oracle_target_projection"
    assert result.test_features.shape == (int(np.sum(target_mask)), 2)


def test_oracle_alignment_can_use_metadata_anchors_without_target_labels():
    features, labels, subjects = _rotated_subject_features(seed=113)
    source_mask = subjects != "s2"
    target_mask = subjects == "s2"
    anchor_values = np.asarray([f"event-{label}" for label in labels], dtype=object)
    config = source_alignment_config(
        method="mcca",
        anchor_mode="event_code_mean",
        components=2,
        target_projection=ORACLE_TARGET_CALIBRATED_ALIGNMENT,
    )

    result = align_train_test_features(
        train_features=features[source_mask],
        train_labels=labels[source_mask],
        train_subject_ids=subjects[source_mask],
        train_anchor_values=anchor_values[source_mask],
        test_features=features[target_mask],
        target_anchor_values=anchor_values[target_mask],
        config=config,
    )

    assert result.metadata["alignment_target_projection"] == ORACLE_TARGET_CALIBRATED_ALIGNMENT
    assert result.metadata["alignment_target_projection_fit"] == "oracle_target_calibrated_template_ridge_least_squares"
    assert result.metadata["alignment_target_labels_used"] is False
    assert result.metadata["alignment_target_anchor_values_used"] is True
    assert result.metadata["alignment_debug_upper_bound"] is True
    assert result.metadata["alignment_valid_for_benchmark"] is False
    assert result.diagnostics["target_transform_type"] == "oracle_target_calibrated_template_ridge_least_squares"
