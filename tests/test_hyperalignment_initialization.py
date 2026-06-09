import numpy as np
import pytest

from neureptrace.decoding.hyperalignment_initialization import (
    HYPERALIGNMENT_INITIALIZATION_MODES,
    class_alignment_matrices,
    fit_class_hyperalignment,
    fit_hyperalignment,
    fit_projection_to_hyperalignment,
    transform_with_projection,
)


def _aligned_subjects():
    rng = np.random.default_rng(0)
    return {
        "s1": rng.normal(size=(6, 4)),
        "s2": rng.normal(size=(6, 4)) + 0.1,
        "s3": rng.normal(size=(6, 4)) - 0.1,
    }


def _assert_orthonormal_columns(matrix: np.ndarray) -> None:
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(matrix.shape[1]), atol=1e-10)


def test_mean_initialized_hyperalignment_fits_common_space():
    aligned = _aligned_subjects()

    model = fit_hyperalignment(aligned, n_components=3, n_iterations=2, initialization="mean")

    assert HYPERALIGNMENT_INITIALIZATION_MODES == ("pca", "mean")
    assert model.n_components == 3
    assert model.template.shape == (6, 3)
    assert model.group_feature_mean.shape == (4,)
    assert model.group_projection.shape == (4, 3)
    _assert_orthonormal_columns(model.group_projection)
    assert model.transform("s1", aligned["s1"]).shape == (6, 3)


def test_pca_initialized_group_projection_is_orthonormalized():
    aligned = _aligned_subjects()

    model = fit_hyperalignment(aligned, n_components=3, n_iterations=2, initialization="pca")

    assert model.group_projection.shape == (4, 3)
    _assert_orthonormal_columns(model.group_projection)


def test_class_hyperalignment_accepts_mean_initialization():
    aligned = _aligned_subjects()
    features = {subject: np.vstack([matrix, matrix + 0.01]) for subject, matrix in aligned.items()}
    labels = {subject: np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]) for subject in aligned}

    model, alignment = fit_class_hyperalignment(
        features,
        labels,
        sample_mode="class_repetition",
        n_repetitions_per_class=2,
        n_components=3,
        n_iterations=2,
        initialization="mean",
    )

    assert alignment.sample_mode == "class_repetition"
    assert alignment.n_repetitions_per_class == 2
    assert alignment.repetition_selection == "random"
    assert alignment.repetition_seed == 0
    assert model.template.shape == (6, 3)
    _assert_orthonormal_columns(model.group_projection)


def test_hyperalignment_class_repetition_uses_common_offsets_when_counts_differ():
    features = {
        "a": np.array([[0.0], [1.0], [2.0], [3.0], [100.0], [101.0], [102.0], [103.0]]),
        "b": np.array([[10.0], [11.0], [12.0], [13.0], [14.0], [110.0], [111.0], [112.0], [113.0], [114.0]]),
    }
    labels = {
        "a": np.array([1, 1, 1, 1, 2, 2, 2, 2]),
        "b": np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
    }

    alignment = class_alignment_matrices(features, labels, sample_mode="class_repetition", n_repetitions_per_class=2)

    assert alignment.repetition_offsets_by_class is not None
    assert np.allclose(
        alignment.aligned_by_subject["b"].ravel() - alignment.aligned_by_subject["a"].ravel(),
        np.full(alignment.aligned_by_subject["a"].shape[0], 10.0),
    )


def test_target_hyperalignment_projection_preserves_nonzero_template_mean() -> None:
    features = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    template_offset = np.array([10.0, -3.0])
    template = features @ rotation + template_offset

    projection = fit_projection_to_hyperalignment(features, template=template)
    transformed = transform_with_projection(features, projection)

    np.testing.assert_allclose(np.mean(transformed, axis=0), np.mean(template, axis=0), atol=1e-10)
    np.testing.assert_allclose(
        transformed - np.mean(transformed, axis=0, keepdims=True),
        template - np.mean(template, axis=0, keepdims=True),
        atol=1e-10,
    )


def test_hyperalignment_caps_components_to_common_centered_rank():
    subjects = {
        "s1": np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        "s2": np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]),
    }

    model = fit_hyperalignment(subjects, n_components=8, n_iterations=2)

    assert model.n_components == 1
    assert model.template.shape == (3, 1)


def test_hyperalignment_rejects_zero_centered_rank():
    subjects = {
        "s1": np.ones((3, 2)),
        "s2": np.ones((3, 2)) * 2.0,
    }

    with pytest.raises(ValueError, match="after centering"):
        fit_hyperalignment(subjects, n_components=8, n_iterations=2)


def test_class_hyperalignment_alignment_exposes_rank_warning():
    features = {
        "s1": np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0], [1.0, 1.0], [2.0, 2.0]]),
        "s2": np.array([[1.1, 0.0], [2.1, 0.0], [0.0, 1.1], [0.0, 2.1], [1.1, 1.1], [2.1, 2.1]]),
    }
    labels = {subject: np.array([0, 0, 1, 1, 2, 2]) for subject in features}

    model, alignment = fit_class_hyperalignment(
        features,
        labels,
        sample_mode="class_mean",
        n_components=64,
        n_iterations=2,
    )

    assert alignment.n_alignment_rows == 3
    assert alignment.n_classes == 3
    assert alignment.max_centered_rank == 2
    assert "class_mean" in str(alignment.low_rank_warning)
    assert model.n_components == 2


def test_pca_initialization_still_allows_different_feature_dimensions():
    rng = np.random.default_rng(1)

    model = fit_hyperalignment({"s1": rng.normal(size=(6, 4)), "s2": rng.normal(size=(6, 5))}, n_components=3, n_iterations=2)

    assert model.n_components == 3
    assert model.group_feature_mean is None
    assert model.group_projection is None


def test_mean_initialization_requires_matching_feature_dimensions():
    rng = np.random.default_rng(2)

    with pytest.raises(ValueError, match="same feature dimension"):
        fit_hyperalignment({"s1": rng.normal(size=(6, 4)), "s2": rng.normal(size=(6, 5))}, n_components=3, n_iterations=2, initialization="mean")


def test_unknown_hyperalignment_initialization_rejected():
    with pytest.raises(ValueError, match="Unsupported hyperalignment initialization"):
        fit_hyperalignment(_aligned_subjects(), initialization="unsupported")
