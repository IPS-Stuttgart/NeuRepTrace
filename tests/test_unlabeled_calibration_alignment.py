import numpy as np
import pytest

from neureptrace.decoding.unlabeled_calibration_alignment import (
    CATEGORY2_UNLABELED_CALIBRATION_ALIGNMENT,
    align_train_test_with_unlabeled_calibration,
    unlabeled_calibration_alignment_config,
)


def _calibration_protocol_fixture(seed=123):
    rng = np.random.default_rng(seed)
    n_subjects = 3
    n_features = 5
    n_calibration_anchors = 8
    labels_one_subject = np.repeat(np.arange(3), 5)
    task_prototypes = np.array(
        [
            [2.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 2.0, 0.5, 0.0, 0.5],
            [0.5, 0.0, 2.0, 0.0, 0.5],
        ]
    )
    calibration_prototypes = rng.normal(size=(n_calibration_anchors, n_features))
    calibration_anchors_one_subject = np.asarray([f"movie-frame-{index:02d}" for index in range(n_calibration_anchors)], dtype=object)

    train_rows = []
    train_labels = []
    train_subjects = []
    calibration_rows = []
    calibration_subjects = []
    calibration_anchors = []
    for subject in range(n_subjects):
        q, _r = np.linalg.qr(rng.normal(size=(n_features, n_features)))
        train_rows.append(task_prototypes[labels_one_subject] @ q + 0.01 * rng.normal(size=(labels_one_subject.size, n_features)))
        train_labels.append(labels_one_subject)
        train_subjects.extend([f"s{subject}"] * labels_one_subject.size)
        calibration_rows.append(calibration_prototypes @ q + 0.01 * rng.normal(size=calibration_prototypes.shape))
        calibration_subjects.extend([f"s{subject}"] * n_calibration_anchors)
        calibration_anchors.extend(calibration_anchors_one_subject.tolist())

    return {
        "features": np.vstack(train_rows),
        "labels": np.concatenate(train_labels),
        "subjects": np.asarray(train_subjects, dtype=object),
        "calibration_features": np.vstack(calibration_rows),
        "calibration_subjects": np.asarray(calibration_subjects, dtype=object),
        "calibration_anchors": np.asarray(calibration_anchors, dtype=object),
    }


@pytest.mark.parametrize(
    ("method", "target_transform_type"),
    [
        ("procrustes", "unlabeled_calibration_template_procrustes"),
        ("hyperalignment", "unlabeled_calibration_template_procrustes"),
        ("mcca", "unlabeled_calibration_template_ridge_least_squares"),
    ],
)
def test_unlabeled_calibration_alignment_uses_separate_target_calibration_anchors(method, target_transform_type):
    data = _calibration_protocol_fixture()
    source_mask = data["subjects"] != "s2"
    target_mask = data["subjects"] == "s2"
    source_calibration_mask = data["calibration_subjects"] != "s2"
    target_calibration_mask = data["calibration_subjects"] == "s2"

    result = align_train_test_with_unlabeled_calibration(
        train_features=data["features"][source_mask],
        train_labels=data["labels"][source_mask],
        train_subject_ids=data["subjects"][source_mask],
        test_features=data["features"][target_mask],
        source_calibration_features=data["calibration_features"][source_calibration_mask],
        source_calibration_subject_ids=data["calibration_subjects"][source_calibration_mask],
        source_calibration_anchor_values=data["calibration_anchors"][source_calibration_mask],
        target_calibration_features=data["calibration_features"][target_calibration_mask],
        target_calibration_anchor_values=data["calibration_anchors"][target_calibration_mask],
        config=unlabeled_calibration_alignment_config(method=method, components=2),
    )

    assert result.train_features.shape == (int(np.sum(source_mask)), 2)
    assert result.test_features.shape == (int(np.sum(target_mask)), 2)
    assert result.metadata["alignment_protocol"] == CATEGORY2_UNLABELED_CALIBRATION_ALIGNMENT
    assert result.metadata["alignment_target_projection"] == CATEGORY2_UNLABELED_CALIBRATION_ALIGNMENT
    assert result.metadata["alignment_uses_unlabeled_target_data"] is True
    assert result.metadata["alignment_target_labels_used"] is False
    assert result.metadata["alignment_target_pseudo_labels_used"] is False
    assert result.metadata["alignment_target_anchor_values_used"] is True
    assert result.metadata["alignment_unlabeled_target_calibrated"] is True
    assert result.metadata["alignment_target_calibrated"] is False
    assert result.metadata["alignment_oracle_target_calibrated"] is False
    assert result.metadata["alignment_debug_upper_bound"] is False
    assert result.metadata["alignment_valid_for_benchmark"] is True
    assert result.metadata["alignment_valid_for_strict_source_only"] is False
    assert result.metadata["alignment_source_calibration_rows"] == int(np.sum(source_calibration_mask))
    assert result.metadata["alignment_target_calibration_rows"] == int(np.sum(target_calibration_mask))
    assert result.metadata["alignment_target_alignment_rows"] == 8
    assert "category-2" in result.metadata["alignment_protocol_note"]
    assert result.diagnostics["target_transform_type"] == target_transform_type
    assert result.diagnostics["uses_unlabeled_target_data"] is True


def test_unlabeled_calibration_alignment_requires_source_calibration_for_each_source_subject():
    data = _calibration_protocol_fixture(seed=321)
    source_mask = data["subjects"] != "s2"
    target_mask = data["subjects"] == "s2"
    source_calibration_mask = data["calibration_subjects"] != "s1"
    target_calibration_mask = data["calibration_subjects"] == "s2"

    with pytest.raises(ValueError, match="Every source decoding subject must have calibration-run rows"):
        align_train_test_with_unlabeled_calibration(
            train_features=data["features"][source_mask],
            train_labels=data["labels"][source_mask],
            train_subject_ids=data["subjects"][source_mask],
            test_features=data["features"][target_mask],
            source_calibration_features=data["calibration_features"][source_calibration_mask],
            source_calibration_subject_ids=data["calibration_subjects"][source_calibration_mask],
            source_calibration_anchor_values=data["calibration_anchors"][source_calibration_mask],
            target_calibration_features=data["calibration_features"][target_calibration_mask],
            target_calibration_anchor_values=data["calibration_anchors"][target_calibration_mask],
            config=unlabeled_calibration_alignment_config(method="hyperalignment", components=2),
        )


def test_unlabeled_calibration_alignment_rejects_missing_target_calibration_anchors():
    data = _calibration_protocol_fixture(seed=456)
    source_mask = data["subjects"] != "s2"
    target_mask = data["subjects"] == "s2"
    source_calibration_mask = data["calibration_subjects"] != "s2"
    target_calibration_mask = data["calibration_subjects"] == "s2"
    target_anchors = data["calibration_anchors"][target_calibration_mask].copy()
    target_anchors[0] = ""

    with pytest.raises(ValueError, match="target_calibration_anchor_values contains missing"):
        align_train_test_with_unlabeled_calibration(
            train_features=data["features"][source_mask],
            train_labels=data["labels"][source_mask],
            train_subject_ids=data["subjects"][source_mask],
            test_features=data["features"][target_mask],
            source_calibration_features=data["calibration_features"][source_calibration_mask],
            source_calibration_subject_ids=data["calibration_subjects"][source_calibration_mask],
            source_calibration_anchor_values=data["calibration_anchors"][source_calibration_mask],
            target_calibration_features=data["calibration_features"][target_calibration_mask],
            target_calibration_anchor_values=target_anchors,
            config=unlabeled_calibration_alignment_config(method="mcca", components=2),
        )
