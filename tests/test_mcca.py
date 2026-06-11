import numpy as np
import pytest

from neureptrace.decoding.mcca import class_alignment_matrices, fit_class_mcca, fit_mcca
from neureptrace.decoding.mcca_target import class_alignment_matrix, fit_target_mcca_projection


def _synthetic_subjects(seed=13):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(24, 3))
    subjects = {}
    for subject in range(4):
        mixing = rng.normal(size=(3, 8))
        subjects[subject] = latent @ mixing + 0.05 * rng.normal(size=(24, 8))
    return subjects


def _object_label_vector(values):
    vector = np.empty(len(values), dtype=object)
    vector[:] = list(values)
    return vector


def test_fit_mcca_recovers_shared_rows():
    subjects = _synthetic_subjects()
    model = fit_mcca(subjects, n_components=3, regularization=1e-5)

    transformed = [model.transform(subject, features) for subject, features in subjects.items()]
    pairwise = []
    for left in range(len(transformed)):
        for right in range(left + 1, len(transformed)):
            score = np.corrcoef(transformed[left][:, 0], transformed[right][:, 0])[0, 1]
            pairwise.append(abs(score))

    assert model.n_components == 3
    assert np.mean(pairwise) > 0.8
    assert model.group_projection is not None
    assert model.transform_group(subjects[0]).shape == (24, 3)


def test_mcca_group_projection_rescale_uses_subject_specific_centering():
    """The calibration-free group projection should be scaled on centered subjects.

    A source-average fallback projection is applied to an unseen target after
    centering the target. Its scale should therefore be estimated from source
    matrices centered with their own fitted means, not from source matrices
    centered with the across-source average mean. The latter can inject large
    between-subject offsets into the scale estimate and shrink held-out features.
    """

    rng = np.random.default_rng(31)
    latent = rng.normal(size=(18, 3))
    mixing = rng.normal(size=(3, 7))
    base = latent @ mixing
    subjects = {
        "low_offset": base - 100.0 + 0.01 * rng.normal(size=base.shape),
        "mid_offset": base + 20.0 + 0.01 * rng.normal(size=base.shape),
        "high_offset": base + 150.0 + 0.01 * rng.normal(size=base.shape),
    }

    model = fit_mcca(subjects, n_components=3, regularization=1e-5, normalize_components=True)
    transformed = [
        model.transform_group(matrix, feature_mean=np.mean(matrix, axis=0))
        for matrix in subjects.values()
    ]

    np.testing.assert_allclose(
        np.std(np.vstack(transformed), axis=0, ddof=1),
        np.ones(model.n_components),
        atol=1e-10,
    )


def test_fit_mcca_caps_requested_components_to_centered_row_rank():
    base = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    subjects = {
        "a": base,
        "b": base
        @ np.array(
            [
                [1.0, 0.2, 0.0, 0.0],
                [0.0, 1.0, 0.3, 0.0],
                [0.1, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        "c": base
        @ np.array(
            [
                [0.8, 0.0, 0.2, 0.0],
                [0.1, 1.1, 0.0, 0.0],
                [0.0, 0.4, 0.9, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    }

    model = fit_mcca(subjects, n_components=8, regularization=1e-6)

    assert model.n_components == 2
    assert model.transform("a", subjects["a"]).shape == (3, 2)


def test_class_alignment_matrices_class_mean():
    features = {
        "a": np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]]),
        "b": np.array([[2.0, 1.0], [4.0, 1.0], [1.0, 3.0], [1.0, 5.0]]),
    }
    labels = {"a": np.array([1, 1, 2, 2]), "b": np.array([1, 1, 2, 2])}

    alignment = class_alignment_matrices(features, labels, sample_mode="class_mean")

    assert alignment.classes.tolist() == [1, 2]
    assert alignment.n_repetitions_per_class is None
    assert alignment.repetition_selection is None
    assert np.allclose(alignment.aligned_by_subject["a"], [[2.0, 0.0], [0.0, 3.0]])
    assert alignment.n_alignment_rows == 2
    assert alignment.n_classes == 2
    assert alignment.max_centered_rank == 1
    assert "class_mean" in str(alignment.low_rank_warning)


def test_class_alignment_matrix_uses_explicit_class_order():
    features = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
    labels = np.array([1, 1, 2, 2])

    aligned = class_alignment_matrix(features, labels, classes=np.array([2, 1]), sample_mode="class_mean")

    assert np.allclose(aligned, [[0.0, 3.0], [2.0, 0.0]])
    with pytest.raises(ValueError, match="absent"):
        class_alignment_matrix(features, labels, classes=np.array([1, 3]), sample_mode="class_mean")


def test_target_class_alignment_matrix_preserves_mixed_label_order():
    features = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0]])
    labels = np.array([1, "stim-b", 1, "stim-b"], dtype=object)

    aligned = class_alignment_matrix(features, labels, sample_mode="class_mean")

    np.testing.assert_allclose(aligned, [[2.0, 0.0], [0.0, 3.0]])


def test_class_alignment_matrices_accept_tuple_object_anchor_labels():
    features = {
        "a": np.array([[1.0], [3.0], [10.0], [30.0]]),
        "b": np.array([[101.0], [103.0], [110.0], [130.0]]),
    }
    labels = {
        "a": _object_label_vector(
            [
                ("face", "famous"),
                ("face", "famous"),
                ("face", "scrambled"),
                ("face", "scrambled"),
            ]
        ),
        "b": _object_label_vector(
            [
                ("face", "famous"),
                ("face", "famous"),
                ("face", "scrambled"),
                ("face", "scrambled"),
            ]
        ),
    }

    mean_alignment = class_alignment_matrices(features, labels, sample_mode="class_mean")

    assert mean_alignment.classes.tolist() == [("face", "famous"), ("face", "scrambled")]
    np.testing.assert_allclose(mean_alignment.aligned_by_subject["a"], [[2.0], [20.0]])
    np.testing.assert_allclose(mean_alignment.aligned_by_subject["b"], [[102.0], [120.0]])

    repetition_alignment = class_alignment_matrices(
        features,
        labels,
        sample_mode="class_repetition",
        n_repetitions_per_class=2,
        repetition_selection="first",
    )
    np.testing.assert_allclose(repetition_alignment.aligned_by_subject["a"].ravel(), [1.0, 3.0, 10.0, 30.0])


def test_class_alignment_matrices_class_repetition_defaults_to_seeded_random():
    features = {
        "a": np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
        "b": np.array([[11.0], [12.0], [13.0], [14.0], [15.0], [16.0]]),
    }
    labels = {"a": np.array([1, 2, 1, 2, 1, 2]), "b": np.array([1, 2, 1, 2, 1, 2])}

    alignment = class_alignment_matrices(features, labels, sample_mode="class_repetition", n_repetitions_per_class=2)
    repeated = class_alignment_matrices(features, labels, sample_mode="class_repetition", n_repetitions_per_class=2)

    assert alignment.n_repetitions_per_class == 2
    assert alignment.repetition_selection == "random"
    assert alignment.repetition_seed == 0
    assert alignment.aligned_by_subject["a"].ravel().tolist() == [3.0, 5.0, 4.0, 6.0]
    assert repeated.aligned_by_subject["a"].ravel().tolist() == alignment.aligned_by_subject["a"].ravel().tolist()


def test_class_alignment_matrices_class_repetition_allows_legacy_first_selection():
    features = {
        "a": np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
        "b": np.array([[11.0], [12.0], [13.0], [14.0], [15.0], [16.0]]),
    }
    labels = {"a": np.array([1, 2, 1, 2, 1, 2]), "b": np.array([1, 2, 1, 2, 1, 2])}

    alignment = class_alignment_matrices(
        features,
        labels,
        sample_mode="class_repetition",
        n_repetitions_per_class=2,
        repetition_selection="first",
    )

    assert alignment.repetition_selection == "first"
    assert alignment.aligned_by_subject["a"].ravel().tolist() == [1.0, 3.0, 2.0, 4.0]


def test_class_alignment_matrices_class_repetition_uses_common_offsets_when_counts_differ():
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
    assert alignment.selected_offsets_by_class is not None
    assert set(alignment.selected_offsets_by_class) == {0, 1}
    assert np.allclose(
        alignment.aligned_by_subject["b"].ravel() - alignment.aligned_by_subject["a"].ravel(),
        np.full(alignment.aligned_by_subject["a"].shape[0], 10.0),
    )


def test_target_class_alignment_matrix_can_reuse_source_selected_offsets_when_counts_differ():
    source_features = {
        "a": np.array([[0.0], [1.0], [2.0], [3.0], [100.0], [101.0], [102.0], [103.0]]),
        "b": np.array([[10.0], [11.0], [12.0], [13.0], [14.0], [110.0], [111.0], [112.0], [113.0], [114.0]]),
    }
    source_labels = {
        "a": np.array([1, 1, 1, 1, 2, 2, 2, 2]),
        "b": np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
    }
    source_alignment = class_alignment_matrices(
        source_features,
        source_labels,
        sample_mode="class_repetition",
        n_repetitions_per_class=2,
    )
    assert source_alignment.selected_offsets_by_class is not None
    target_features = np.array(
        [
            [1000.0],
            [1001.0],
            [1002.0],
            [1003.0],
            [1004.0],
            [2000.0],
            [2001.0],
            [2002.0],
            [2003.0],
            [2004.0],
        ]
    )
    target_labels = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

    aligned = class_alignment_matrix(
        target_features,
        target_labels,
        classes=source_alignment.classes,
        sample_mode="class_repetition",
        n_repetitions_per_class=source_alignment.n_repetitions_per_class,
        repetition_selection=source_alignment.repetition_selection or "random",
        repetition_seed=source_alignment.repetition_seed,
        selected_offsets_by_class=source_alignment.selected_offsets_by_class,
    )
    expected = np.vstack(
        [
            target_features[target_labels == class_label][source_alignment.selected_offsets_by_class[class_position]]
            for class_position, class_label in enumerate(source_alignment.classes)
        ]
    )

    np.testing.assert_allclose(aligned, expected)
    with pytest.raises(ValueError, match="must match"):
        class_alignment_matrix(
            target_features,
            target_labels,
            classes=source_alignment.classes,
            sample_mode="class_repetition",
            n_repetitions_per_class=1,
            selected_offsets_by_class=source_alignment.selected_offsets_by_class,
        )


def test_target_class_alignment_matrix_reuses_repetition_sampling_options():
    features = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    labels = np.array([1, 2, 1, 2, 1, 2])

    default_aligned = class_alignment_matrix(features, labels, sample_mode="class_repetition", n_repetitions_per_class=2)
    first_aligned = class_alignment_matrix(
        features,
        labels,
        sample_mode="class_repetition",
        n_repetitions_per_class=2,
        repetition_selection="first",
    )

    assert default_aligned.ravel().tolist() == [3.0, 5.0, 4.0, 6.0]
    assert first_aligned.ravel().tolist() == [1.0, 3.0, 2.0, 4.0]


def test_fit_class_mcca_rejects_missing_class():
    features = {"a": np.ones((4, 2)), "b": np.ones((4, 2))}
    labels = {"a": np.array([0, 0, 1, 1]), "b": np.array([0, 0, 0, 0])}

    with pytest.raises(ValueError, match="expected"):
        fit_class_mcca(features, labels)


def test_fit_mcca_rejects_alignment_with_no_shared_rank():
    subjects = {
        "a": np.ones((3, 4)),
        "b": np.ones((3, 4)) * 2.0,
    }

    with pytest.raises(ValueError, match="no retained centered components"):
        fit_mcca(subjects, n_components=4, regularization=1e-6)


def test_fit_mcca_rejects_one_degenerate_subject_anchor_set():
    rng = np.random.default_rng(7)
    subjects = {
        "flat": np.ones((5, 4)),
        "structured_a": rng.normal(size=(5, 4)),
        "structured_b": rng.normal(size=(5, 4)),
    }

    with pytest.raises(ValueError, match="subject 'flat'.*no retained centered components"):
        fit_mcca(subjects, n_components=4, regularization=1e-6)


def test_fit_mcca_caps_components_to_centered_shared_rank():
    rng = np.random.default_rng(42)
    latent = rng.normal(size=(3, 2))
    features = {}
    for subject in range(3):
        mixing = rng.normal(size=(2, 6))
        features[subject] = latent @ mixing + 0.01 * rng.normal(size=(3, 6))

    model = fit_mcca(features, n_components=64, regularization=1e-5)

    assert model.n_components == 2
    assert model.component_scores.shape == (3, 2)
    assert all(projection.projection.shape[1] == 2 for projection in model.projections.values())


def test_fit_mcca_rejects_fractional_component_counts():
    features = {
        "a": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        "b": np.array([[0.0, 0.0], [0.9, 0.1], [0.1, 1.0]]),
    }

    with pytest.raises(ValueError, match="integer component count"):
        fit_mcca(features, n_components=0.95)


def test_fit_mcca_group_projection_keeps_normalized_scale_after_averaging():
    rng = np.random.default_rng(123)
    latent = rng.normal(size=(18, 4))
    subjects = {}
    for subject in range(5):
        mixing = rng.normal(size=(4, 12))
        shift = rng.normal(size=(12,)) * 0.5
        subjects[subject] = latent @ mixing + shift + 0.02 * rng.normal(size=(18, 12))

    model = fit_mcca(
        subjects,
        n_components=4,
        regularization=1e-5,
        normalize_components=True,
    )
    pooled_group = np.vstack([model.transform_group(features) for features in subjects.values()])

    np.testing.assert_allclose(
        np.std(pooled_group, axis=0, ddof=1),
        np.ones(model.n_components),
        rtol=0.06,
        atol=0.06,
    )


def test_fit_target_mcca_projection_projects_held_out_subject_to_template():
    subjects = _synthetic_subjects()
    training_subjects = {subject: features for subject, features in subjects.items() if subject != 3}
    model = fit_mcca(training_subjects, n_components=3, regularization=1e-5)

    projection = fit_target_mcca_projection(subjects[3], model, regularization=1e-5)
    transformed = projection.transform(subjects[3])

    correlations = [
        abs(np.corrcoef(transformed[:, component], model.component_scores[:, component])[0, 1])
        for component in range(model.n_components)
    ]
    assert projection.projection.shape == (8, 3)
    assert transformed.shape == model.component_scores.shape
    assert np.mean(correlations) > 0.8


def test_fit_target_mcca_projection_rejects_row_mismatch():
    with pytest.raises(ValueError, match="template rows"):
        fit_target_mcca_projection(np.ones((3, 2)), np.ones((4, 2)), regularization=1e-6)
