import numpy as np
import pytest

import neureptrace  # noqa: F401  Ensures runtime compatibility patches are installed.
from neureptrace.decoding.source_alignment import (
    ORACLE_TARGET_CALIBRATED_ALIGNMENT,
    align_train_test_features,
    source_alignment_config,
)


def _rotated_subject_features(seed=13, n_subjects=4):
    rng = np.random.default_rng(seed)
    labels_one_subject = np.repeat(np.arange(3), 8)
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
        subject_features = prototypes[labels_one_subject] @ q + 0.02 * rng.normal(size=(labels_one_subject.size, 4))
        features.append(subject_features)
        labels.append(labels_one_subject)
        subjects.extend([f"s{subject}"] * labels_one_subject.size)
    return np.vstack(features), np.concatenate(labels), np.asarray(subjects, dtype=object)


@pytest.mark.parametrize(
    ("method", "expected_fit"),
    [
        ("procrustes", "oracle_target_template_procrustes"),
        ("hyperalignment", "oracle_target_template_procrustes"),
        ("mcca", "oracle_target_template_ridge_least_squares"),
    ],
)
def test_oracle_target_alignment_reports_explicit_oracle_projection_fit(method, expected_fit):
    features, labels, subjects = _rotated_subject_features()
    source_mask = subjects != "s3"
    target_mask = subjects == "s3"

    result = align_train_test_features(
        train_features=features[source_mask],
        train_labels=labels[source_mask],
        train_subject_ids=subjects[source_mask],
        test_features=features[target_mask],
        target_labels=labels[target_mask],
        config=source_alignment_config(
            method=method,
            components=2,
            target_projection=ORACLE_TARGET_CALIBRATED_ALIGNMENT,
        ),
    )

    assert result.metadata["alignment_target_projection"] == ORACLE_TARGET_CALIBRATED_ALIGNMENT
    assert result.metadata["alignment_target_projection_fit"] == expected_fit
    assert result.diagnostics["target_transform_type"] == expected_fit
    assert result.metadata["alignment_debug_upper_bound"] is True
    assert result.metadata["alignment_valid_for_benchmark"] is False
    assert result.metadata["alignment_valid_for_strict_source_only"] is False
    assert result.metadata["alignment_target_labels_used"] is True
    assert result.metadata["alignment_target_anchor_values_used"] is True
    assert result.metadata["alignment_oracle_target_projection_source"] == "scored_heldout_target_rows"
    assert result.metadata["alignment_oracle_target_raw_rows_used"] == int(np.count_nonzero(target_mask))
    assert result.metadata["alignment_protocol_note"].startswith("uses scored held-out target labels or anchors")
    assert result.diagnostics["alignment_protocol"] == ORACLE_TARGET_CALIBRATED_ALIGNMENT
    assert result.diagnostics["source_inner_validation_type"] == (
        "source_loso_nearest_centroid_oracle_target_projection_same_rows"
    )


def test_oracle_metadata_anchor_alignment_does_not_claim_target_labels_used():
    features, labels, subjects = _rotated_subject_features(seed=17)
    anchors = np.asarray([f"stim-{label}" for label in labels], dtype=object)
    source_mask = subjects != "s3"
    target_mask = subjects == "s3"

    result = align_train_test_features(
        train_features=features[source_mask],
        train_labels=labels[source_mask],
        train_subject_ids=subjects[source_mask],
        train_anchor_values=anchors[source_mask],
        test_features=features[target_mask],
        target_anchor_values=anchors[target_mask],
        config=source_alignment_config(
            method="mcca",
            anchor_mode="stimulus_id_mean",
            components=2,
            target_projection=ORACLE_TARGET_CALIBRATED_ALIGNMENT,
        ),
    )

    assert result.metadata["alignment_target_projection_fit"] == "oracle_target_template_ridge_least_squares"
    assert result.diagnostics["target_transform_type"] == "oracle_target_template_ridge_least_squares"
    assert result.metadata["alignment_target_labels_used"] is False
    assert result.metadata["alignment_target_anchor_values_used"] is True
    assert result.metadata["alignment_oracle_target_labels_or_anchors_used"] is True
    assert result.metadata["alignment_debug_upper_bound"] is True
    assert result.metadata["alignment_valid_for_benchmark"] is False
