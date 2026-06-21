import numpy as np
import pytest

from neureptrace.decoding.source_alignment import (
    PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT,
    _fit_source_alignment_model,
    _transform_inner_heldout_subject,
    source_alignment_config,
)


def _rotated_subject_features(seed: int = 17, n_subjects: int = 4):
    rng = np.random.default_rng(seed)
    labels_one_subject = np.repeat(np.arange(3), 4)
    prototypes = np.array(
        [
            [2.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 2.0, 0.5, 0.0, 0.0],
            [0.5, 0.0, 2.0, 0.0, 0.5],
        ]
    )
    features = []
    labels = []
    subjects = []
    for subject in range(n_subjects):
        q, _r = np.linalg.qr(rng.normal(size=(5, 5)))
        subject_features = prototypes[labels_one_subject] @ q + 0.02 * rng.normal(size=(labels_one_subject.size, 5))
        features.append(subject_features)
        labels.append(labels_one_subject)
        subjects.extend([f"s{subject}"] * labels_one_subject.size)
    return np.vstack(features), np.concatenate(labels), np.asarray(subjects, dtype=object)


@pytest.mark.parametrize("method", ["procrustes", "hyperalignment", "mcca"])
def test_pseudo_label_source_inner_target_projection_uses_disjoint_calibration_rows(method):
    features, labels, subjects = _rotated_subject_features()
    source_subjects = tuple(f"s{subject}" for subject in range(3))
    features_by_subject = {
        subject: features[subjects == subject]
        for subject in source_subjects
    }
    anchors_by_subject = {
        subject: labels[subjects == subject]
        for subject in source_subjects
    }
    config = source_alignment_config(
        method=method,
        anchor_mode="class_repetition",
        repetition_cap=4,
        components=2,
        target_projection=PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT,
        target_calibration_per_anchor=1,
        target_calibration_seed=23,
    )

    fit = _fit_source_alignment_model(
        features_by_subject,
        anchors_by_subject,
        config=config,
        sample_mode="class_repetition",
        external_anchor_mode=False,
    )
    test_mask = subjects == "s3"

    transformed, evaluation_mask = _transform_inner_heldout_subject(
        test_features=features[test_mask],
        test_anchors=labels[test_mask],
        fit=fit,
        config=config,
    )

    # One pseudo-calibration row per class must be withheld from scoring.  Before
    # the runtime patch, pseudo-label source-inner diagnostics fitted the target
    # projection on all held-out rows and scored those same rows, making the
    # diagnostic oracle-like and inconsistent with the outer calibrated protocol.
    assert fit.alignment.n_repetitions_per_class == 1
    assert evaluation_mask.shape == (12,)
    assert int(np.sum(~evaluation_mask)) == 3
    assert int(np.sum(evaluation_mask)) == 9
    assert transformed.shape[0] == int(np.sum(evaluation_mask))
