from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_alignment import (
    ORACLE_TARGET_CALIBRATED_ALIGNMENT,
    align_train_test_features,
    source_alignment_anchor_availability,
    source_alignment_config,
)


def _composite_anchor_fixture():
    features = []
    labels = []
    subjects = []
    anchors = []
    prototypes = {
        ("run-01", "stim-a"): np.array([1.0, 0.0, 0.0]),
        ("run-01", "stim-b"): np.array([0.0, 1.0, 0.0]),
    }
    for subject_index, subject in enumerate(["s0", "s1", "s2"]):
        offset = np.array([subject_index * 0.05, -subject_index * 0.03, subject_index * 0.02])
        for anchor, prototype in prototypes.items():
            features.append(prototype + offset)
            labels.append(anchor[1])
            subjects.append(subject)
            anchors.append(anchor)
    return (
        np.asarray(features, dtype=float),
        np.asarray(labels, dtype=object),
        np.asarray(subjects, dtype=object),
        anchors,
    )


def test_source_alignment_preserves_composite_train_anchor_values() -> None:
    features, labels, subjects, anchors = _composite_anchor_fixture()

    result = align_train_test_features(
        train_features=features,
        train_labels=labels,
        train_subject_ids=subjects,
        train_anchor_values=anchors,
        test_features=features[:2],
        config=source_alignment_config(method="mcca", anchor_mode="stimulus_id_mean", components=1),
        compute_source_inner_diagnostics=False,
    )

    assert result.train_features.shape == (6, 1)
    assert result.test_features.shape == (2, 1)
    assert result.metadata["alignment_anchor_value_source"] == "metadata"
    assert result.metadata["alignment_common_anchor_count"] == 2
    assert result.metadata["alignment_anchor_rows_used"] == 6


def test_source_alignment_preserves_numpy_composite_decoder_labels() -> None:
    features, labels, subjects, _anchors = _composite_anchor_fixture()
    composite_labels = np.array(
        [("run-01", label) for label in labels.tolist()],
        dtype=object,
    )

    result = align_train_test_features(
        train_features=features,
        train_labels=composite_labels,
        train_subject_ids=subjects,
        test_features=features[:2],
        config=source_alignment_config(method="mcca", anchor_mode="class_mean", components=1),
        compute_source_inner_diagnostics=False,
    )

    assert result.train_features.shape == (6, 1)
    assert result.test_features.shape == (2, 1)
    assert result.metadata["alignment_anchor_value_source"] == "decoder_labels"
    assert result.metadata["alignment_common_anchor_count"] == 2


def test_anchor_availability_preserves_numpy_composite_decoder_labels() -> None:
    _features, labels, subjects, _anchors = _composite_anchor_fixture()
    composite_labels = np.array(
        [("run-01", label) for label in labels.tolist()],
        dtype=object,
    )

    row = source_alignment_anchor_availability(
        train_labels=composite_labels,
        train_subject_ids=subjects,
        config=source_alignment_config(method="mcca", anchor_mode="class_mean"),
    )

    assert row["prefit_status"] == "ok"
    assert row["source_anchor_value_source"] == "decoder_labels"
    assert row["n_common_source_anchors"] == 2


def test_anchor_availability_preserves_composite_oracle_target_anchor_values() -> None:
    features, labels, subjects, anchors = _composite_anchor_fixture()

    row = source_alignment_anchor_availability(
        train_labels=labels,
        train_subject_ids=subjects,
        train_anchor_values=anchors,
        target_anchor_values=list(anchors[:2]),
        config=source_alignment_config(
            method="mcca",
            anchor_mode="stimulus_id_mean",
            target_projection=ORACLE_TARGET_CALIBRATED_ALIGNMENT,
        ),
    )

    assert row["prefit_status"] == "ok"
    assert row["n_common_source_anchors"] == 2
    assert row["target_missing_common_anchor_count"] == 0
    assert row["n_target_anchor_values"] == 2


def test_anchor_availability_counts_missing_composite_target_anchors_as_one_anchor() -> None:
    _features, labels, subjects, anchors = _composite_anchor_fixture()

    row = source_alignment_anchor_availability(
        train_labels=labels,
        train_subject_ids=subjects,
        train_anchor_values=anchors,
        target_anchor_values=[anchors[0]],
        config=source_alignment_config(
            method="mcca",
            anchor_mode="stimulus_id_mean",
            target_projection=ORACLE_TARGET_CALIBRATED_ALIGNMENT,
        ),
    )

    assert row["prefit_status"] == "likely_fit_failure"
    assert row["n_common_source_anchors"] == 2
    assert row["n_target_anchor_values"] == 1
    assert row["target_missing_common_anchor_count"] == 1
    assert "stim-b" in row["target_missing_common_anchor_values_preview"]
