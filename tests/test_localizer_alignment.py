import numpy as np
import pytest

from neureptrace.decoding.localizer_alignment import (
    apply_procrustes_transform,
    class_pattern_matrix,
    common_label_values,
    fit_source_localizer_procrustes,
    fit_source_only_procrustes_template,
    fit_target_localizer_procrustes,
    permuted_labels,
)


def _orthogonal(rng, n_features):
    matrix = rng.normal(size=(n_features, n_features))
    left, _s, right_t = np.linalg.svd(matrix, full_matrices=False)
    return left @ right_t


def _localizer_subjects(seed=0, *, n_classes=5, n_features=4, repetitions=8):
    rng = np.random.default_rng(seed)
    template = rng.normal(size=(n_classes, n_features))
    labels = np.repeat(np.arange(1, n_classes + 1), repetitions)
    features_by_subject = {}
    labels_by_subject = {}
    transforms = {}
    for subject_id in ("s1", "s2", "s3", "target"):
        rotation = _orthogonal(rng, n_features)
        center = rng.normal(size=n_features)
        inverse_rows = []
        for class_index in range(n_classes):
            latent = template[class_index] + 0.03 * rng.normal(size=(repetitions, n_features))
            inverse_rows.append((latent - center) @ rotation.T)
        features_by_subject[subject_id] = np.vstack(inverse_rows)
        labels_by_subject[subject_id] = labels.copy()
        transforms[subject_id] = (rotation, center)
    return template, features_by_subject, labels_by_subject, transforms


def test_source_and_target_localizer_alignment_recover_class_patterns():
    _template, features, labels, _transforms = _localizer_subjects()
    source_features = {key: features[key] for key in ("s1", "s2", "s3")}
    source_labels = {key: labels[key] for key in source_features}

    model = fit_source_localizer_procrustes(source_features, source_labels, n_iterations=3)
    target = fit_target_localizer_procrustes(model, features["target"], labels["target"], subject_id="target")

    assert model.source_subject_ids == ("s1", "s2", "s3")
    assert model.classes.tolist() == [1, 2, 3, 4, 5]
    assert target.subject_id == "target"
    for subject_id in source_features:
        aligned = model.transform_source(subject_id, source_features[subject_id])
        patterns = class_pattern_matrix(aligned, source_labels[subject_id], classes=model.classes)
        assert np.mean(np.linalg.norm(patterns - model.template, axis=1)) < 0.08
    aligned_target = target.transform_features(features["target"])
    target_patterns = class_pattern_matrix(aligned_target, labels["target"], classes=model.classes)
    assert np.mean(np.linalg.norm(target_patterns - model.template, axis=1)) < 0.08


def test_block_size_alignment_applies_transform_to_repeated_feature_blocks():
    _template, features, labels, _transforms = _localizer_subjects(seed=2, n_classes=4, n_features=3, repetitions=5)
    source_features = {}
    for subject_id in ("s1", "s2"):
        source_features[subject_id] = np.hstack([features[subject_id], features[subject_id] + 0.01])
    source_labels = {key: labels[key] for key in source_features}

    model = fit_source_localizer_procrustes(source_features, source_labels, block_size=3, n_iterations=2)
    aligned = model.transform_source("s1", source_features["s1"])
    direct = apply_procrustes_transform(source_features["s1"], model.transforms["s1"], block_size=3)
    patterns = class_pattern_matrix(aligned, source_labels["s1"], classes=model.classes, block_size=3)

    assert aligned.shape == source_features["s1"].shape
    assert np.allclose(aligned, direct)
    assert np.mean(np.linalg.norm(patterns - model.template, axis=1)) < 0.1


def test_source_only_template_supports_mean_initialization():
    rng = np.random.default_rng(4)
    matrices = {
        "a": rng.normal(size=(4, 3)),
        "b": rng.normal(size=(4, 3)),
    }

    template = fit_source_only_procrustes_template(matrices, n_iterations=1, initial_template="mean")

    assert template.subject_ids == ("a", "b")
    assert template.initial_template == "mean"
    assert template.template.shape == (4, 3)


def test_common_labels_preserve_first_subject_order_and_target_shuffle_is_seeded():
    labels = {
        "s1": np.array([3, 1, 2, 3, 1, 2]),
        "s2": np.array([2, 1, 3, 2, 1, 3]),
    }

    first = permuted_labels(labels["s1"], seed=7, context=("target",))
    second = permuted_labels(labels["s1"], seed=7, context=("target",))

    assert common_label_values(labels).tolist() == [3, 1, 2]
    assert first.tolist() == second.tolist()
    assert sorted(first.tolist()) == sorted(labels["s1"].tolist())


def test_target_alignment_rejects_missing_localizer_class():
    _template, features, labels, _transforms = _localizer_subjects(seed=5)
    source_features = {key: features[key] for key in ("s1", "s2")}
    source_labels = {key: labels[key] for key in source_features}
    model = fit_source_localizer_procrustes(source_features, source_labels)
    target_labels = labels["target"].copy()
    target_labels[target_labels == 5] = 4

    with pytest.raises(ValueError, match="absent"):
        model.fit_target(features["target"], target_labels)
