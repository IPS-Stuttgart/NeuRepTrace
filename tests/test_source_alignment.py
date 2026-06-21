import numpy as np
import pytest

from neureptrace.decoding.hyperalignment_initialization import fit_class_hyperalignment
from neureptrace.decoding.mcca import fit_class_mcca
from neureptrace.decoding.mcca_target import class_alignment_matrix, fit_target_mcca_projection
from neureptrace.decoding.source_alignment import (
    GROUP_PROJECTION_TARGET_CENTERED,
    _inner_target_calibration_mask,
    ORACLE_TARGET_CALIBRATED_ALIGNMENT,
    PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT,
    TARGET_CALIBRATED_ALIGNMENT,
    _target_alignment_matrix,
    _transform_unsupervised_covariance_alignment_by_subject,
    align_train_test_features,
    normalize_source_alignment_method,
    source_alignment_anchor_availability,
    source_alignment_config,
)


def _rotated_subject_features(seed=13, n_subjects=3):
    rng = np.random.default_rng(seed)
    labels_one_subject = np.repeat(np.arange(3), 8)
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
    for subject in range(n_subjects):
        q, _r = np.linalg.qr(rng.normal(size=(4, 4)))
        subject_features = prototypes[labels_one_subject] @ q + 0.02 * rng.normal(size=(labels_one_subject.size, 4))
        features.append(subject_features)
        labels.append(labels_one_subject)
        subjects.extend([f"s{subject}"] * labels_one_subject.size)
    return np.vstack(features), np.concatenate(labels), np.asarray(subjects, dtype=object)


def _mean_subject_class_distance(features, labels, subjects):
    means = []
    for subject in np.unique(subjects):
        subject_means = []
        for label in np.unique(labels):
            mask = (subjects == subject) & (labels == label)
            subject_means.append(features[mask].mean(axis=0))
        means.append(np.vstack(subject_means))
    distances = []
    for left in range(len(means)):
        for right in range(left + 1, len(means)):
            distances.append(float(np.linalg.norm(means[left] - means[right])))
    return float(np.mean(distances))


def _mean_source_target_class_distance(source_features, source_labels, target_features, target_labels):
    distances = []
    for label in np.unique(source_labels):
        source_mean = source_features[source_labels == label].mean(axis=0)
        target_mean = target_features[target_labels == label].mean(axis=0)
        distances.append(float(np.linalg.norm(source_mean - target_mean)))
    return float(np.mean(distances))


def test_source_alignment_config_rejects_fractional_integer_parameters():
    with pytest.raises(ValueError, match="alignment_repetition_cap must be an integer"):
        source_alignment_config(method="procrustes", repetition_cap=1.5)

    with pytest.raises(ValueError, match="alignment_target_calibration_per_anchor must be an integer"):
        source_alignment_config(
            method="procrustes",
            target_projection=TARGET_CALIBRATED_ALIGNMENT,
            target_calibration_per_anchor=1.5,
        )

    with pytest.raises(ValueError, match="alignment_target_calibration_seed must be an integer"):
        source_alignment_config(
            method="procrustes",
            target_projection=TARGET_CALIBRATED_ALIGNMENT,
            target_calibration_seed=13.5,
        )

    with pytest.raises(ValueError, match="alignment_hyperalignment_iterations must be an integer"):
        source_alignment_config(method="hyperalignment", hyperalignment_iterations=1.5)


def test_source_alignment_config_rejects_bool_numeric_parameters():
    with pytest.raises(ValueError, match="alignment_components must be an integer"):
        source_alignment_config(method="procrustes", components=True)

    with pytest.raises(ValueError, match="alignment_target_calibration_seed must be an integer"):
        source_alignment_config(
            method="procrustes",
            target_projection=TARGET_CALIBRATED_ALIGNMENT,
            target_calibration_seed=True,
        )

    with pytest.raises(ValueError, match="alignment_mcca_regularization"):
        source_alignment_config(method="mcca", mcca_regularization=True)


def test_none_alignment_reproduces_raw_features():
    train_features = np.array([[1.0, 0.0], [0.0, 1.0]])
    test_features = np.array([[0.5, 0.5]])

    result = align_train_test_features(
        train_features=train_features,
        train_labels=np.array([0, 1]),
        train_subject_ids=np.array(["a", "b"]),
        test_features=test_features,
        config=source_alignment_config(method="none"),
    )

    np.testing.assert_allclose(result.train_features, train_features)
    np.testing.assert_allclose(result.test_features, test_features)
    assert result.metadata["alignment_method"] == "none"


def test_metadata_anchor_values_accept_mixed_hashable_types():
    train_features = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.0, 1.1, 0.0],
            [0.9, 0.1, 0.0],
            [0.1, 0.9, 0.0],
        ]
    )
    train_labels = np.array([0, 1, 0, 1, 0, 1])
    train_subjects = np.array(["s0", "s0", "s1", "s1", "s2", "s2"], dtype=object)
    train_anchors = np.array([1, "stim-b", 1, "stim-b", 1, "stim-b"], dtype=object)

    result = align_train_test_features(
        train_features=train_features,
        train_labels=train_labels,
        train_subject_ids=train_subjects,
        train_anchor_values=train_anchors,
        test_features=train_features[:2],
        config=source_alignment_config(
            method="mcca",
            anchor_mode="stimulus_id_mean",
            components=1,
        ),
        compute_source_inner_diagnostics=False,
    )

    assert result.train_features.shape == (6, 1)
    assert result.test_features.shape == (2, 1)
    assert result.metadata["alignment_anchor_value_source"] == "metadata"
    assert result.metadata["alignment_common_anchor_count"] == 2
    assert result.metadata["alignment_anchor_rows_dropped"] == 0


def test_stimulus_repetition_alignment_accepts_mixed_hashable_anchor_types():
    train_features = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.1, 0.0],
            [1.2, 0.0, 0.0],
            [1.3, 0.0, 0.0],
            [0.0, 1.2, 0.0],
            [0.0, 1.3, 0.0],
            [0.9, 0.1, 0.0],
            [1.0, 0.1, 0.0],
            [0.1, 0.9, 0.0],
            [0.1, 1.0, 0.0],
        ]
    )
    train_labels = np.array([0, 0, 1, 1] * 3)
    train_subjects = np.array(["s0"] * 4 + ["s1"] * 4 + ["s2"] * 4, dtype=object)
    train_anchors = np.array([1, 1, "stim-b", "stim-b"] * 3, dtype=object)

    result = align_train_test_features(
        train_features=train_features,
        train_labels=train_labels,
        train_subject_ids=train_subjects,
        train_anchor_values=train_anchors,
        test_features=train_features[:4],
        config=source_alignment_config(method="mcca", anchor_mode="stimulus_id_repetition", components=1),
        compute_source_inner_diagnostics=True,
    )

    assert result.train_features.shape == (12, 1)
    assert result.metadata["alignment_common_anchor_count"] == 2
    assert result.metadata["alignment_repetitions_per_class"] == 2
    assert np.isfinite(result.diagnostics["source_inner_raw_balanced_accuracy"])


def test_strict_alignment_rejects_target_labels():
    with pytest.raises(ValueError, match="target labels"):
        align_train_test_features(
            train_features=np.ones((4, 2)),
            train_labels=np.array([0, 1, 0, 1]),
            train_subject_ids=np.array(["a", "a", "b", "b"]),
            test_features=np.ones((2, 2)),
            target_labels=np.array([0, 1]),
            config=source_alignment_config(method="procrustes"),
        )


def test_strict_alignment_rejects_target_anchor_values():
    with pytest.raises(ValueError, match="target anchor values"):
        align_train_test_features(
            train_features=np.ones((4, 2)),
            train_labels=np.array([0, 1, 0, 1]),
            train_subject_ids=np.array(["a", "a", "b", "b"]),
            test_features=np.ones((2, 2)),
            train_anchor_values=np.array(["stim-a", "stim-b", "stim-a", "stim-b"]),
            target_anchor_values=np.array(["stim-a", "stim-b"]),
            config=source_alignment_config(method="procrustes", anchor_mode="stimulus_id_mean"),
        )


@pytest.mark.parametrize(
    ("alias", "method"),
    [
        ("euclidean_alignment", "euclidean"),
        ("coral_alignment", "coral"),
        ("target_covariance_alignment", "target_baseline_covariance"),
        ("sensor_covariance_normalization", "subject_sensor_covariance"),
    ],
)
def test_unsupervised_alignment_method_aliases(alias, method):
    assert normalize_source_alignment_method(alias) == method


def test_alignment_times_same_decode_window_metadata():
    config = source_alignment_config(method="procrustes", times="same_decode_window")
    metadata = config.static_metadata()

    assert config.same_decode_window is True
    assert config.times == ()
    assert metadata["alignment_times"] == "same_decode_window"
    assert metadata["alignment_window_mode"] == "same_decode_window"
    assert metadata["alignment_same_decode_window"] is True


def test_alignment_defaults_to_same_decode_window_until_fixed_windows_are_implemented():
    config = source_alignment_config(method="procrustes")
    metadata = config.static_metadata()

    assert config.same_decode_window is True
    assert metadata["alignment_times"] == "same_decode_window"
    assert metadata["alignment_window_mode"] == "same_decode_window"


def test_alignment_times_accept_metadata_pipe_roundtrip():
    config = source_alignment_config(
        method="procrustes",
        times="0.088|0.136|0.184",
    )
    metadata = config.static_metadata()

    assert config.same_decode_window is False
    assert config.times == (0.088, 0.136, 0.184)
    assert metadata["alignment_times"] == "0.088|0.136|0.184"


def test_oracle_alignment_requires_target_labels():
    train_features, train_labels, train_subjects = _rotated_subject_features()
    with pytest.raises(ValueError, match="requires held-out target labels"):
        align_train_test_features(
            train_features=train_features,
            train_labels=train_labels,
            train_subject_ids=train_subjects,
            test_features=train_features[:6],
            config=source_alignment_config(
                method="procrustes",
                target_projection=ORACLE_TARGET_CALIBRATED_ALIGNMENT,
            ),
        )


@pytest.mark.parametrize("method", ["procrustes", "hyperalignment", "mcca"])
def test_source_alignment_methods_expose_group_projection_metadata(method):
    train_features, train_labels, train_subjects = _rotated_subject_features()
    raw_distance = _mean_subject_class_distance(train_features, train_labels, train_subjects)

    result = align_train_test_features(
        train_features=train_features,
        train_labels=train_labels,
        train_subject_ids=train_subjects,
        test_features=train_features[:6],
        config=source_alignment_config(method=method, components=2),
    )

    aligned_distance = _mean_subject_class_distance(result.train_features, train_labels, train_subjects)
    assert result.train_features.shape == (train_features.shape[0], 2)
    assert result.test_features.shape == (6, 2)
    assert result.metadata["alignment_method"] == method
    assert result.metadata["alignment_target_projection"] == "group_projection"
    assert result.metadata["alignment_uses_class_labels"] is True
    assert result.metadata["alignment_n_components"] == 2
    assert result.diagnostics["alignment_method"] == method
    assert result.diagnostics["sample_mode"] == "class_mean"
    assert result.diagnostics["n_source_subjects"] == 3
    assert result.diagnostics["n_classes"] == 3
    assert result.diagnostics["n_alignment_rows"] == 3
    assert result.diagnostics["requested_components"] == 2
    assert result.diagnostics["actual_components"] == 2
    assert result.diagnostics["feature_dim"] == train_features.shape[1]
    assert result.diagnostics["decode_feature_dim"] == 2
    assert result.diagnostics["uses_channel_projection_collapse"] is False
    assert result.diagnostics["alignment_dimensionality_reduction"] is True
    assert "class_mean" in result.diagnostics["alignment_low_rank_warning"]
    assert result.metadata["alignment_low_rank_warning"] == result.diagnostics["alignment_low_rank_warning"]
    assert np.isfinite(result.diagnostics["anchor_row_correlation_before"])
    assert np.isfinite(result.diagnostics["anchor_row_correlation_after"])
    assert np.isfinite(result.diagnostics["source_inner_decoding_before_alignment"])
    assert np.isfinite(result.diagnostics["source_inner_decoding_after_alignment"])
    assert result.diagnostics["source_inner_raw_balanced_accuracy"] == result.diagnostics["source_inner_decoding_before_alignment"]
    assert result.diagnostics["source_inner_aligned_balanced_accuracy"] == result.diagnostics["source_inner_decoding_after_alignment"]
    assert np.isfinite(result.diagnostics["source_inner_aligned_minus_raw"])
    assert result.diagnostics["source_inner_validation_type"] == "strict_source_loso_nearest_centroid_group_projection"
    assert result.diagnostics["target_transform_type"] == "source_group_projection"
    assert aligned_distance < raw_distance


def test_source_inner_diagnostics_skip_raw_when_aligned_inner_folds_fail(monkeypatch):
    """Raw/aligned source-inner diagnostics must score the same inner folds."""

    import neureptrace.decoding.source_alignment as source_alignment_module

    features, labels, subjects = _rotated_subject_features(seed=67)
    original_fit = source_alignment_module._fit_source_alignment_model
    full_outer_subject_count = len(np.unique(subjects))

    def fail_inner_alignment(*args, **kwargs):
        features_by_subject = args[0] if args else kwargs["features_by_subject"]
        if len(features_by_subject) < full_outer_subject_count:
            raise ValueError("simulated inner alignment failure")
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(source_alignment_module, "_fit_source_alignment_model", fail_inner_alignment)

    result = align_train_test_features(
        train_features=features,
        train_labels=labels,
        train_subject_ids=subjects,
        test_features=features[:6],
        config=source_alignment_config(method="mcca", components=2),
        compute_source_inner_diagnostics=True,
    )

    assert result.train_features.shape == (features.shape[0], 2)
    assert result.diagnostics["source_inner_raw_balanced_accuracy"] == ""
    assert result.diagnostics["source_inner_aligned_balanced_accuracy"] == ""
    assert result.diagnostics["source_inner_aligned_minus_raw"] == ""


@pytest.mark.parametrize("method", ["procrustes", "hyperalignment", "mcca"])
def test_target_centered_group_projection_uses_unlabeled_target_mean(method):
    features, labels, subjects = _rotated_subject_features(seed=53)
    source_mask = subjects != "s2"
    target_mask = subjects == "s2"
    shifted_target = features[target_mask] + np.array([25.0, -10.0, 5.0, 3.0])
    base_kwargs = {
        "train_features": features[source_mask],
        "train_labels": labels[source_mask],
        "train_subject_ids": subjects[source_mask],
        "test_features": shifted_target,
    }

    strict = align_train_test_features(
        **base_kwargs,
        config=source_alignment_config(method=method, components=2),
    )
    centered = align_train_test_features(
        **base_kwargs,
        config=source_alignment_config(
            method=method,
            components=2,
            target_projection=GROUP_PROJECTION_TARGET_CENTERED,
        ),
    )

    assert centered.metadata["alignment_target_projection"] == GROUP_PROJECTION_TARGET_CENTERED
    assert centered.metadata["alignment_strict_source_only"] is False
    assert centered.metadata["alignment_uses_unlabeled_target_data"] is True
    assert centered.metadata["alignment_valid_for_benchmark"] is False
    assert centered.metadata["alignment_valid_for_strict_source_only"] is False
    assert "unlabeled target feature mean" in centered.metadata["alignment_protocol_note"]
    assert centered.diagnostics["uses_unlabeled_target_data"] is True
    assert centered.diagnostics["target_transform_type"] == "source_group_projection_target_centered"
    assert (
        centered.diagnostics["source_inner_validation_type"]
        == "strict_source_loso_nearest_centroid_group_projection_target_centered"
    )
    assert not np.allclose(strict.test_features, centered.test_features)


def test_inner_target_calibration_mask_keeps_disjoint_eval_rows_per_anchor():
    anchors = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c"], dtype=object)

    mask = _inner_target_calibration_mask(
        anchors,
        classes=np.array(["a", "b", "c"], dtype=object),
        per_anchor=1,
        seed=13,
    )

    assert mask.dtype == bool
    assert int(mask.sum()) == 3
    for anchor in ("a", "b", "c"):
        anchor_mask = anchors == anchor
        assert int(np.sum(mask & anchor_mask)) == 1
        assert int(np.sum(~mask & anchor_mask)) == 2


def test_inner_target_calibration_mask_rejects_when_eval_would_be_empty():
    anchors = np.array(["a", "b", "c"], dtype=object)

    with pytest.raises(ValueError, match="disjoint from scored rows"):
        _inner_target_calibration_mask(anchors, classes=np.array(["a", "b", "c"], dtype=object), per_anchor=1, seed=13)


def test_metadata_anchor_values_reject_numpy_missing_values():
    train_features = np.arange(12, dtype=float).reshape(6, 2)
    train_labels = np.array([0, 1, 2, 0, 1, 2])
    train_subjects = np.array(["s0", "s0", "s0", "s1", "s1", "s1"], dtype=object)
    train_anchors = np.array(["a", "b", "c", "a", np.float64("nan"), "c"], dtype=object)

    with pytest.raises(ValueError, match="missing values"):
        align_train_test_features(
            train_features=train_features,
            train_labels=train_labels,
            train_subject_ids=train_subjects,
            train_anchor_values=train_anchors,
            test_features=train_features[:3],
            config=source_alignment_config(method="mcca", anchor_mode="stimulus_id_mean"),
        )


@pytest.mark.parametrize("missing_anchor", ["", "   ", "NA", "nan", "<NA>", "None", "null", "NaT"])
def test_metadata_anchor_values_reject_text_missing_values(missing_anchor):
    train_features = np.arange(12, dtype=float).reshape(6, 2)
    train_labels = np.array([0, 1, 2, 0, 1, 2])
    train_subjects = np.array(["s0", "s0", "s0", "s1", "s1", "s1"], dtype=object)
    train_anchors = np.array(["a", "b", "c", "a", missing_anchor, "c"], dtype=object)

    with pytest.raises(ValueError, match="missing values"):
        align_train_test_features(
            train_features=train_features,
            train_labels=train_labels,
            train_subject_ids=train_subjects,
            train_anchor_values=train_anchors,
            test_features=train_features[:3],
            config=source_alignment_config(method="mcca", anchor_mode="stimulus_id_mean"),
        )


def test_anchor_availability_reports_missing_train_anchor_values_without_raising():
    row = source_alignment_anchor_availability(
        train_labels=np.array([0, 1, 0, 1]),
        train_subject_ids=np.array(["s0", "s0", "s1", "s1"]),
        train_anchor_values=np.array(["stim-a", "", "stim-a", "stim-b"], dtype=object),
        config=source_alignment_config(method="mcca", anchor_mode="stimulus_id_mean"),
    )

    assert row["prefit_status"] == "likely_fit_failure"
    assert "invalid_train_anchor_values" in row["prefit_failure_reason"]
    assert "missing values" in row["prefit_failure_detail"]


def test_anchor_availability_flags_missing_oracle_target_anchor_values():
    row = source_alignment_anchor_availability(
        train_labels=np.array([0, 1, 0, 1, 0, 1]),
        train_subject_ids=np.array(["s0", "s0", "s0", "s1", "s1", "s1"]),
        train_anchor_values=np.array(["stim-a", "stim-b", "stim-c", "stim-a", "stim-b", "stim-c"], dtype=object),
        target_anchor_values=np.array(["stim-a", "", "stim-c"], dtype=object),
        config=source_alignment_config(
            method="mcca",
            anchor_mode="stimulus_id_mean",
            target_projection=ORACLE_TARGET_CALIBRATED_ALIGNMENT,
        ),
    )

    assert row["prefit_status"] == "likely_fit_failure"
    assert "target_projection_contains_missing_anchor_values" in row["prefit_failure_reason"]
    assert row["target_missing_common_anchor_count"] == 1


@pytest.mark.parametrize(
    ("method", "target_transform_type"),
    [
        ("euclidean", "unlabeled_target_covariance_whitening"),
        ("coral", "unlabeled_target_covariance_recoloring"),
        ("target_baseline_covariance", "unlabeled_target_pooled_source_to_target_covariance"),
        ("subject_sensor_covariance", "unlabeled_target_covariance_whitening"),
    ],
)
def test_unsupervised_covariance_alignment_uses_no_class_anchors(method, target_transform_type):
    train_features, train_labels, train_subjects = _rotated_subject_features(seed=29)
    target_features = train_features[train_subjects == "s2"] * np.array([1.5, 0.5, 1.2, 0.8])

    result = align_train_test_features(
        train_features=train_features[train_subjects != "s2"],
        train_labels=train_labels[train_subjects != "s2"],
        train_subject_ids=train_subjects[train_subjects != "s2"],
        test_features=target_features,
        config=source_alignment_config(method=method),
    )

    assert result.train_features.shape[1] == train_features.shape[1]
    assert result.test_features.shape == target_features.shape
    assert result.metadata["alignment_anchor_value_source"] == "unlabeled_covariance"
    assert result.metadata["alignment_uses_unlabeled_target_data"] is True
    assert result.metadata["alignment_uses_class_labels"] is False
    assert result.metadata["alignment_valid_for_benchmark"] is False
    assert result.metadata["alignment_valid_for_strict_source_only"] is False
    assert "unlabeled target covariance" in result.metadata["alignment_protocol_note"]
    assert result.metadata["alignment_target_labels_used"] is False
    assert result.diagnostics["sample_mode"] == "unlabeled_covariance"
    assert result.diagnostics["uses_unlabeled_target_data"] is True
    assert result.diagnostics["target_transform_type"] == target_transform_type
    assert result.diagnostics["covariance_alignment_estimator"] == "full"
    assert np.isfinite(result.diagnostics["source_inner_raw_balanced_accuracy"])


def test_unsupervised_covariance_alignment_preserves_interleaved_train_row_order():
    train_features, train_labels, train_subjects = _rotated_subject_features(seed=31)
    subject_ids = tuple(dict.fromkeys(train_subjects.tolist()))
    per_subject_positions = [np.flatnonzero(train_subjects == subject_id) for subject_id in subject_ids]
    interleaved_order = np.ravel(np.column_stack(per_subject_positions))
    train_features = train_features[interleaved_order]
    train_subjects = train_subjects[interleaved_order]
    test_features = train_features[:5]

    result = align_train_test_features(
        train_features=train_features,
        train_labels=train_labels[interleaved_order],
        train_subject_ids=train_subjects,
        test_features=test_features,
        config=source_alignment_config(method="euclidean"),
    )

    features_by_subject = {subject_id: train_features[train_subjects == subject_id] for subject_id in subject_ids}
    transformed_by_subject, _test_transformed, _metadata = _transform_unsupervised_covariance_alignment_by_subject(
        features_by_subject,
        test_features,
        method="euclidean",
    )
    expected = np.empty_like(result.train_features)
    for subject_id in subject_ids:
        expected[train_subjects == subject_id] = transformed_by_subject[subject_id]

    np.testing.assert_allclose(result.train_features, expected)
    grouped = np.vstack([transformed_by_subject[subject_id] for subject_id in subject_ids])
    assert not np.allclose(result.train_features, grouped)


def test_unsupervised_covariance_alignment_rejects_oracle_target_projection():
    with pytest.raises(ValueError, match="does not support target-calibrated projections"):
        source_alignment_config(method="coral", target_projection=ORACLE_TARGET_CALIBRATED_ALIGNMENT)


@pytest.mark.parametrize("method", ["procrustes", "hyperalignment", "mcca"])
def test_stimulus_anchor_values_are_distinct_from_decoder_labels(method):
    train_features, train_labels, train_subjects = _rotated_subject_features(seed=11)
    repetitions = np.tile(np.tile(np.arange(8), 3), 3)
    train_anchors = np.asarray([f"stim-{label}-{rep % 2}" for label, rep in zip(train_labels, repetitions, strict=True)])

    result = align_train_test_features(
        train_features=train_features,
        train_labels=train_labels,
        train_subject_ids=train_subjects,
        train_anchor_values=train_anchors,
        test_features=train_features[:6],
        config=source_alignment_config(
            method=method,
            anchor_mode="stimulus_id_mean",
            anchor_column="stim_file",
            components=2,
        ),
    )

    assert result.train_features.shape == (train_features.shape[0], 2)
    assert result.metadata["alignment_anchor_mode"] == "stimulus_id_mean"
    assert result.metadata["alignment_anchor_column"] == "stim_file"
    assert result.metadata["alignment_anchor_value_source"] == "metadata"
    assert result.metadata["alignment_uses_class_labels"] is False
    assert result.metadata["alignment_common_anchor_count"] == 6
    assert result.metadata["alignment_anchor_rows_dropped"] == 0
    assert result.metadata["alignment_target_anchor_values_used"] is False


def test_anchor_availability_flags_no_common_stimulus_anchors():
    row = source_alignment_anchor_availability(
        train_labels=np.array([0, 1, 0, 1]),
        train_subject_ids=np.array(["s0", "s0", "s1", "s1"]),
        train_anchor_values=np.array(["stim-a", "stim-b", "stim-c", "stim-d"], dtype=object),
        config=source_alignment_config(method="mcca", anchor_mode="stimulus_id_mean"),
    )

    assert row["prefit_status"] == "likely_fit_failure"
    assert "no_common_source_alignment_anchors" in row["prefit_failure_reason"]
    assert row["n_common_source_anchors"] == 0
    assert row["source_anchor_rows_retained"] == 0
    assert row["source_anchor_rows_dropped"] == 4
    assert row["estimated_alignment_rows"] == 0


@pytest.mark.parametrize("method", ["procrustes", "hyperalignment", "mcca"])
def test_anchor_availability_flags_too_few_alignment_rows_for_all_class_methods(method):
    row = source_alignment_anchor_availability(
        train_labels=np.array([0, 1, 0, 1]),
        train_subject_ids=np.array(["s0", "s0", "s1", "s1"]),
        train_anchor_values=np.array(["shared", "left-only", "shared", "right-only"], dtype=object),
        config=source_alignment_config(method=method, anchor_mode="stimulus_id_mean"),
    )

    assert row["estimated_alignment_rows"] == 1
    assert row["prefit_status"] == "likely_fit_failure"
    assert f"{method}_requires_at_least_two_aligned_rows" in row["prefit_failure_reason"]


def test_anchor_availability_flags_missing_oracle_target_anchors():
    row = source_alignment_anchor_availability(
        train_labels=np.array([0, 1, 0, 1, 0, 1]),
        train_subject_ids=np.array(["s0", "s0", "s0", "s1", "s1", "s1"]),
        train_anchor_values=np.array(["stim-a", "stim-b", "stim-c", "stim-a", "stim-b", "stim-c"], dtype=object),
        target_anchor_values=np.array(["stim-a", "stim-b"], dtype=object),
        config=source_alignment_config(
            method="mcca",
            anchor_mode="stimulus_id_mean",
            target_projection=ORACLE_TARGET_CALIBRATED_ALIGNMENT,
        ),
    )

    assert row["prefit_status"] == "likely_fit_failure"
    assert "target_subject_missing_alignment_anchors" in row["prefit_failure_reason"]
    assert row["n_common_source_anchors"] == 3
    assert row["target_missing_common_anchor_count"] == 1
    assert row["target_missing_common_anchor_values_preview"] == "stim-c"


def test_anchor_availability_flags_insufficient_target_calibration_repetitions():
    row = source_alignment_anchor_availability(
        train_labels=np.array([0, 0, 1, 1, 0, 0, 1, 1]),
        train_subject_ids=np.array(["s0", "s0", "s0", "s0", "s1", "s1", "s1", "s1"]),
        target_calibration_labels=np.array([0, 1]),
        config=source_alignment_config(
            method="mcca",
            anchor_mode="class_repetition",
            repetition_cap=2,
            target_projection=TARGET_CALIBRATED_ALIGNMENT,
            target_calibration_per_anchor=2,
        ),
    )

    assert row["estimated_alignment_rows"] == 4
    assert row["estimated_repetitions_per_anchor"] == 2
    assert row["prefit_status"] == "likely_fit_failure"
    assert "target_calibration_subject_insufficient_alignment_anchor_repetitions" in row["prefit_failure_reason"]
    assert "require at least 2" in row["prefit_failure_detail"]


@pytest.mark.parametrize("method", ["procrustes", "hyperalignment", "mcca"])
def test_oracle_target_calibrated_alignment_is_debug_upper_bound(method):
    features, labels, subjects = _rotated_subject_features(seed=41)
    source_mask = subjects != "s2"
    target_mask = subjects == "s2"
    config_kwargs = {"method": method, "components": 2}

    strict = align_train_test_features(
        train_features=features[source_mask],
        train_labels=labels[source_mask],
        train_subject_ids=subjects[source_mask],
        test_features=features[target_mask],
        config=source_alignment_config(**config_kwargs),
    )
    oracle = align_train_test_features(
        train_features=features[source_mask],
        train_labels=labels[source_mask],
        train_subject_ids=subjects[source_mask],
        test_features=features[target_mask],
        target_labels=labels[target_mask],
        config=source_alignment_config(
            **config_kwargs,
            target_projection=ORACLE_TARGET_CALIBRATED_ALIGNMENT,
        ),
    )

    strict_distance = _mean_source_target_class_distance(
        strict.train_features,
        labels[source_mask],
        strict.test_features,
        labels[target_mask],
    )
    oracle_distance = _mean_source_target_class_distance(
        oracle.train_features,
        labels[source_mask],
        oracle.test_features,
        labels[target_mask],
    )
    assert oracle_distance < strict_distance
    assert oracle.metadata["alignment_target_projection"] == ORACLE_TARGET_CALIBRATED_ALIGNMENT
    assert oracle.metadata["alignment_oracle_target_calibrated"] is True
    assert oracle.metadata["alignment_debug_upper_bound"] is True
    assert oracle.metadata["alignment_valid_for_benchmark"] is False
    assert oracle.metadata["alignment_target_labels_used"] is True
    assert oracle.metadata["alignment_protocol_note"] == "debug upper bound only; not valid for benchmark"


def test_oracle_target_calibrated_alignment_accepts_target_anchor_values():
    features, labels, subjects = _rotated_subject_features(seed=43)
    repetitions = np.tile(np.tile(np.arange(8), 3), 3)
    anchors = np.asarray([f"stim-{label}-{rep % 2}" for label, rep in zip(labels, repetitions, strict=True)])
    source_mask = subjects != "s2"
    target_mask = subjects == "s2"

    oracle = align_train_test_features(
        train_features=features[source_mask],
        train_labels=labels[source_mask],
        train_subject_ids=subjects[source_mask],
        train_anchor_values=anchors[source_mask],
        test_features=features[target_mask],
        target_anchor_values=anchors[target_mask],
        config=source_alignment_config(
            method="procrustes",
            anchor_mode="stimulus_id_mean",
            anchor_column="stim_file",
            components=2,
            target_projection=ORACLE_TARGET_CALIBRATED_ALIGNMENT,
        ),
    )

    assert oracle.metadata["alignment_target_projection"] == ORACLE_TARGET_CALIBRATED_ALIGNMENT
    assert oracle.metadata["alignment_target_labels_used"] is False
    assert oracle.metadata["alignment_target_anchor_values_used"] is True
    assert oracle.metadata["alignment_valid_for_benchmark"] is False


@pytest.mark.parametrize("method", ["procrustes", "hyperalignment", "mcca"])
def test_source_inner_diagnostics_follow_target_projection_for_oracle_alignment(method):
    features, labels, subjects = _rotated_subject_features(seed=71, n_subjects=4)
    source_mask = subjects != "s3"
    target_mask = subjects == "s3"

    oracle = align_train_test_features(
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
        compute_source_inner_diagnostics=True,
    )

    assert oracle.diagnostics["source_inner_validation_type"] == "source_loso_nearest_centroid_target_projection"
    assert np.isfinite(oracle.diagnostics["source_inner_raw_balanced_accuracy"])
    assert np.isfinite(oracle.diagnostics["source_inner_aligned_balanced_accuracy"])


def test_target_class_repetition_alignment_reuses_source_offsets():
    features = np.array(
        [
            [10.0],
            [11.0],
            [12.0],
            [13.0],
            [100.0],
            [101.0],
            [102.0],
            [103.0],
        ]
    )
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    selected_offsets = {0: np.array([1, 3]), 1: np.array([0, 2])}

    aligned = _target_alignment_matrix(
        features,
        labels,
        classes=np.array([0, 1]),
        config=source_alignment_config(
            method="mcca",
            anchor_mode="class_repetition",
            target_projection=ORACLE_TARGET_CALIBRATED_ALIGNMENT,
        ),
        n_repetitions_per_class=2,
        selected_offsets_by_class=selected_offsets,
    )

    np.testing.assert_allclose(aligned, [[11.0], [13.0], [100.0], [102.0]])
    with pytest.raises(ValueError, match="outside"):
        _target_alignment_matrix(
            features,
            labels,
            classes=np.array([0, 1]),
            config=source_alignment_config(
                method="mcca",
                anchor_mode="class_repetition",
                target_projection=ORACLE_TARGET_CALIBRATED_ALIGNMENT,
            ),
            n_repetitions_per_class=2,
            selected_offsets_by_class={0: np.array([1, 9]), 1: np.array([0, 2])},
        )


def test_class_repetition_alignment_uses_shared_common_offsets_for_unequal_counts():
    def make_subject(counts):
        rows = []
        labels = []
        for class_label, count in enumerate(counts):
            for repetition in range(count):
                rows.append([100.0 * class_label + repetition, 10.0 * class_label + repetition])
                labels.append(class_label)
        return np.asarray(rows, dtype=float), np.asarray(labels)

    features_by_subject = {}
    labels_by_subject = {}
    for subject, counts in {"s0": (5, 6), "s1": (7, 5), "s2": (6, 8)}.items():
        features_by_subject[subject], labels_by_subject[subject] = make_subject(counts)

    for fitter in (fit_class_mcca, fit_class_hyperalignment):
        _model, alignment = fitter(
            features_by_subject,
            labels_by_subject,
            sample_mode="class_repetition",
            n_repetitions_per_class=3,
            repetition_seed=7,
            n_components=1,
        )
        offsets_by_class = alignment.selected_offsets_by_class
        assert offsets_by_class is not None

        for subject_id, features in features_by_subject.items():
            labels = labels_by_subject[subject_id]
            expected = []
            for class_position, class_label in enumerate(alignment.classes):
                class_features = features[labels == class_label]
                offsets = offsets_by_class[class_position]
                assert int(np.max(offsets)) < min(
                    int(np.sum(subject_labels == class_label))
                    for subject_labels in labels_by_subject.values()
                )
                expected.extend(class_features[offsets])
            np.testing.assert_allclose(alignment.aligned_by_subject[subject_id], np.vstack(expected))


def test_oracle_class_repetition_alignment_runs_end_to_end_with_shared_offsets():
    features, labels, subjects = _rotated_subject_features(seed=61)
    source_mask = subjects != "s2"
    target_mask = subjects == "s2"

    result = align_train_test_features(
        train_features=features[source_mask],
        train_labels=labels[source_mask],
        train_subject_ids=subjects[source_mask],
        test_features=features[target_mask],
        target_labels=labels[target_mask],
        config=source_alignment_config(
            method="mcca",
            anchor_mode="class_repetition",
            repetition_cap=2,
            components=2,
            target_projection=ORACLE_TARGET_CALIBRATED_ALIGNMENT,
        ),
    )

    assert result.metadata["alignment_target_alignment_rows"] == 6
    assert result.test_features.shape[0] == int(np.sum(target_mask))


@pytest.mark.parametrize(
    ("method", "target_transform_type"),
    [
        ("procrustes", "target_calibrated_template_procrustes"),
        ("hyperalignment", "target_calibrated_template_procrustes"),
        ("mcca", "target_calibrated_template_ridge_least_squares"),
    ],
)
def test_target_calibrated_alignment_uses_separate_calibration_rows(method, target_transform_type):
    features, labels, subjects = _rotated_subject_features(seed=47)
    source_mask = subjects != "s2"
    target_positions = np.flatnonzero(subjects == "s2")
    calibration_positions = np.asarray([target_positions[labels[target_positions] == label][0] for label in np.unique(labels)])
    evaluation_positions = np.asarray([index for index in target_positions if index not in set(calibration_positions.tolist())])

    target_calibrated = align_train_test_features(
        train_features=features[source_mask],
        train_labels=labels[source_mask],
        train_subject_ids=subjects[source_mask],
        test_features=features[evaluation_positions],
        target_calibration_features=features[calibration_positions],
        target_calibration_labels=labels[calibration_positions],
        config=source_alignment_config(
            method=method,
            components=2,
            target_projection=TARGET_CALIBRATED_ALIGNMENT,
            target_calibration_per_anchor=1,
            target_calibration_seed=19,
        ),
    )

    assert target_calibrated.test_features.shape[0] == evaluation_positions.size
    assert target_calibrated.metadata["alignment_target_projection"] == TARGET_CALIBRATED_ALIGNMENT
    assert target_calibrated.metadata["alignment_target_calibrated"] is True
    assert target_calibrated.metadata["alignment_oracle_target_calibrated"] is False
    assert target_calibrated.metadata["alignment_debug_upper_bound"] is False
    assert target_calibrated.metadata["alignment_valid_for_benchmark"] is False
    assert target_calibrated.metadata["alignment_target_alignment_rows"] == 3
    assert target_calibrated.metadata["alignment_target_labels_used"] is True
    assert target_calibrated.metadata["alignment_protocol"] == TARGET_CALIBRATED_ALIGNMENT
    assert target_calibrated.metadata["alignment_protocol_note"] == (
        "uses disjoint target calibration rows; not valid for strict source-only benchmark"
    )
    assert target_calibrated.diagnostics["target_transform_type"] == target_transform_type


@pytest.mark.parametrize(
    ("method", "target_transform_type"),
    [
        ("procrustes", "pseudo_label_template_procrustes"),
        ("hyperalignment", "pseudo_label_template_procrustes"),
        ("mcca", "pseudo_label_template_ridge_least_squares"),
    ],
)
def test_pseudo_label_target_calibrated_alignment_marks_pseudo_labels(method, target_transform_type):
    features, labels, subjects = _rotated_subject_features(seed=53)
    source_mask = subjects != "s2"
    target_positions = np.flatnonzero(subjects == "s2")
    calibration_positions = np.asarray([target_positions[labels[target_positions] == label][0] for label in np.unique(labels)])
    pseudo_labels = labels[calibration_positions]

    result = align_train_test_features(
        train_features=features[source_mask],
        train_labels=labels[source_mask],
        train_subject_ids=subjects[source_mask],
        test_features=features[target_positions],
        target_calibration_features=features[calibration_positions],
        target_calibration_labels=pseudo_labels,
        config=source_alignment_config(
            method=method,
            components=2,
            target_projection=PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT,
        ),
    )

    assert result.metadata["alignment_target_projection"] == PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT
    assert result.metadata["alignment_target_calibrated"] is False
    assert result.metadata["alignment_pseudo_label_target_calibrated"] is True
    assert result.metadata["alignment_target_labels_used"] is False
    assert result.metadata["alignment_target_pseudo_labels_used"] is True
    assert result.metadata["alignment_target_anchor_values_used"] is True
    assert result.metadata["alignment_valid_for_benchmark"] is False
    assert result.metadata["alignment_protocol"] == PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT
    assert result.diagnostics["target_transform_type"] == target_transform_type
    assert result.diagnostics["uses_unlabeled_target_data"] is True


def test_target_calibrated_mcca_uses_public_target_projection_helper():
    features, labels, subjects = _rotated_subject_features(seed=53)
    source_mask = subjects != "s2"
    target_positions = np.flatnonzero(subjects == "s2")
    calibration_positions = np.asarray([target_positions[labels[target_positions] == label][0] for label in np.unique(labels)])
    evaluation_positions = np.asarray([index for index in target_positions if index not in set(calibration_positions.tolist())])
    source_subjects = tuple(dict.fromkeys(subjects[source_mask].tolist()))
    features_by_subject = {subject: features[source_mask][subjects[source_mask] == subject] for subject in source_subjects}
    labels_by_subject = {subject: labels[source_mask][subjects[source_mask] == subject] for subject in source_subjects}

    model, alignment = fit_class_mcca(
        features_by_subject,
        labels_by_subject,
        sample_mode="class_mean",
        n_components=2,
        regularization=1e-6,
    )
    target_anchors = class_alignment_matrix(
        features[calibration_positions],
        labels[calibration_positions],
        classes=alignment.classes,
        sample_mode="class_mean",
    )
    projection = fit_target_mcca_projection(target_anchors, model, regularization=1e-6)
    expected = projection.transform(features[evaluation_positions])

    target_calibrated = align_train_test_features(
        train_features=features[source_mask],
        train_labels=labels[source_mask],
        train_subject_ids=subjects[source_mask],
        test_features=features[evaluation_positions],
        target_calibration_features=features[calibration_positions],
        target_calibration_labels=labels[calibration_positions],
        config=source_alignment_config(method="mcca", components=2, target_projection=TARGET_CALIBRATED_ALIGNMENT),
    )

    np.testing.assert_allclose(target_calibrated.test_features, expected)


def test_target_calibrated_class_repetition_mcca_reuses_source_offsets() -> None:
    features, labels, subjects = _rotated_subject_features(seed=61)
    source_mask = subjects != "s2"
    target_positions = np.flatnonzero(subjects == "s2")
    classes = np.unique(labels)
    calibration_positions = np.concatenate(
        [target_positions[labels[target_positions] == class_label][:2] for class_label in classes]
    )
    evaluation_positions = np.asarray(
        [index for index in target_positions if index not in set(calibration_positions.tolist())]
    )
    source_subjects = tuple(dict.fromkeys(subjects[source_mask].tolist()))
    features_by_subject = {
        subject: features[source_mask][subjects[source_mask] == subject]
        for subject in source_subjects
    }
    labels_by_subject = {
        subject: labels[source_mask][subjects[source_mask] == subject]
        for subject in source_subjects
    }

    model, alignment = fit_class_mcca(
        features_by_subject,
        labels_by_subject,
        sample_mode="class_repetition",
        n_repetitions_per_class=2,
        repetition_selection="first",
        n_components=2,
        regularization=1e-6,
    )
    assert alignment.selected_offsets_by_class is not None
    for offsets in alignment.selected_offsets_by_class.values():
        np.testing.assert_array_equal(offsets, np.array([0, 1]))

    target_anchors = class_alignment_matrix(
        features[calibration_positions],
        labels[calibration_positions],
        classes=alignment.classes,
        sample_mode="class_repetition",
        n_repetitions_per_class=alignment.n_repetitions_per_class,
        selected_offsets_by_class=alignment.selected_offsets_by_class,
    )
    projection = fit_target_mcca_projection(target_anchors, model, regularization=1e-6)
    expected = projection.transform(features[evaluation_positions])

    target_calibrated = align_train_test_features(
        train_features=features[source_mask],
        train_labels=labels[source_mask],
        train_subject_ids=subjects[source_mask],
        test_features=features[evaluation_positions],
        target_calibration_features=features[calibration_positions],
        target_calibration_labels=labels[calibration_positions],
        config=source_alignment_config(
            method="mcca",
            anchor_mode="class_repetition",
            components=2,
            repetition_cap=8,
            target_projection=TARGET_CALIBRATED_ALIGNMENT,
            target_calibration_per_anchor=2,
        ),
    )

    np.testing.assert_allclose(target_calibrated.test_features, expected)


def test_class_repetition_cap_is_capped_by_available_counts():
    train_features, train_labels, train_subjects = _rotated_subject_features()

    result = align_train_test_features(
        train_features=train_features,
        train_labels=train_labels,
        train_subject_ids=train_subjects,
        test_features=train_features[:2],
        config=source_alignment_config(
            method="mcca",
            anchor_mode="class_repetition",
            repetition_cap=99,
            components=4,
        ),
    )

    assert result.metadata["alignment_anchor_mode"] == "class_repetition"
    assert result.metadata["alignment_repetitions_per_class"] == 8


@pytest.mark.parametrize("method", ["procrustes", "hyperalignment", "mcca"])
def test_target_calibrated_class_repetition_respects_calibration_cap(method):
    features, labels, subjects = _rotated_subject_features(seed=59)
    source_mask = subjects != "s2"
    target_positions = np.flatnonzero(subjects == "s2")
    calibration_positions = np.asarray(
        [target_positions[labels[target_positions] == label][0] for label in np.unique(labels)]
    )
    evaluation_positions = np.asarray(
        [index for index in target_positions if index not in set(calibration_positions.tolist())]
    )

    result = align_train_test_features(
        train_features=features[source_mask],
        train_labels=labels[source_mask],
        train_subject_ids=subjects[source_mask],
        test_features=features[evaluation_positions],
        target_calibration_features=features[calibration_positions],
        target_calibration_labels=labels[calibration_positions],
        config=source_alignment_config(
            method=method,
            anchor_mode="class_repetition",
            repetition_cap=8,
            components=2,
            target_projection=TARGET_CALIBRATED_ALIGNMENT,
            target_calibration_per_anchor=1,
            target_calibration_seed=23,
        ),
    )

    assert result.metadata["alignment_repetitions_per_class"] == 1
    assert result.metadata["alignment_target_alignment_rows"] == 3
    assert result.test_features.shape[0] == evaluation_positions.size
