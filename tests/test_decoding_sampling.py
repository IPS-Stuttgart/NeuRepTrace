import numpy as np
import pytest

from neureptrace.decoding import (
    DEFAULT_CLASS_LIMIT_SELECTION,
    normalize_class_limit_seed,
    normalize_class_limit_selection,
    select_class_limited_indices,
)


def test_select_class_limited_indices_keeps_all_when_uncapped():
    labels = np.array([1, 2, 1, 2])

    selected = select_class_limited_indices(labels, None)

    assert selected.tolist() == [0, 1, 2, 3]


def test_select_class_limited_indices_defaults_to_seeded_random_selection():
    labels = np.array([1, 2, 1, 2, 1, 2])

    selected = select_class_limited_indices(labels, 2)
    repeated = select_class_limited_indices(labels, 2)

    assert DEFAULT_CLASS_LIMIT_SELECTION == "random"
    assert selected.tolist() == [1, 2, 4, 5]
    assert repeated.tolist() == selected.tolist()


def test_select_class_limited_indices_first_preserves_input_order():
    labels = np.array([1, 2, 1, 2, 1, 2])

    selected = select_class_limited_indices(labels, 2, selection="first")

    assert selected.tolist() == [0, 1, 2, 3]


def test_select_class_limited_indices_random_is_seeded_by_context():
    labels = np.array([1, 2, 1, 2, 1, 2])

    selected = select_class_limited_indices(labels, 2, selection="random", seed=0, seed_context=1)
    repeated = select_class_limited_indices(labels, 2, selection="random", seed=0, seed_context=1)
    other_context = select_class_limited_indices(labels, 2, selection="random", seed=0, seed_context=2)

    assert selected.tolist() == [1, 2, 3, 4]
    assert repeated.tolist() == selected.tolist()
    assert other_context.tolist() != selected.tolist()


def test_select_class_limited_indices_accepts_mixed_hashable_labels():
    labels = np.array([1, "stim-b", 1, "stim-b", 1, "stim-b"], dtype=object)

    first = select_class_limited_indices(labels, 2, selection="first")
    random = select_class_limited_indices(labels, 2, selection="random", seed=0)
    repeated = select_class_limited_indices(labels, 2, selection="random", seed=0)

    assert first.tolist() == [0, 1, 2, 3]
    assert random.tolist() == repeated.tolist()
    assert len(random) == 4


def test_select_class_limited_indices_treats_tuple_labels_atomically():
    labels = [
        ("face", "early"),
        ("house", "late"),
        ("face", "early"),
        ("house", "late"),
        ("face", "early"),
        ("house", "late"),
    ]

    first = select_class_limited_indices(labels, 2, selection="first")
    random = select_class_limited_indices(labels, 2, selection="random", seed=0)

    assert first.tolist() == [0, 1, 2, 3]
    assert len(random) == 4
    assert np.all(random < len(labels))


def test_select_class_limited_indices_treats_numpy_tuple_label_matrix_atomically():
    labels = np.asarray(
        [
            ("face", "early"),
            ("house", "late"),
            ("face", "early"),
            ("house", "late"),
            ("face", "early"),
            ("house", "late"),
        ],
        dtype=object,
    )

    first = select_class_limited_indices(labels, 2, selection="first")
    random = select_class_limited_indices(labels, 2, selection="random", seed=0)

    assert labels.ndim == 2
    assert first.tolist() == [0, 1, 2, 3]
    assert len(random) == 4
    assert np.all(random < labels.shape[0])


def test_select_class_limited_indices_validates_inputs():
    with pytest.raises(ValueError, match="max_per_class"):
        select_class_limited_indices([1, 2], 0)
    with pytest.raises(ValueError, match="max_per_class"):
        select_class_limited_indices([1, 2], 1.5)
    with pytest.raises(ValueError, match="max_per_class"):
        select_class_limited_indices([1, 2], True)
    with pytest.raises(ValueError, match="selection"):
        select_class_limited_indices([1, 2], 1, selection="middle")
    with pytest.raises(ValueError, match="seed"):
        normalize_class_limit_seed(-1)
    with pytest.raises(ValueError, match="seed"):
        normalize_class_limit_seed(1.5)
    with pytest.raises(ValueError, match="seed"):
        normalize_class_limit_seed(True)
    with pytest.raises(ValueError, match="seed_context"):
        select_class_limited_indices([1, 1, 1], 1, seed_context=1.25)
    with pytest.raises(ValueError, match="seed_context"):
        select_class_limited_indices([1, 1, 1], 1, seed_context=[0, True])


def test_select_class_limited_indices_rejects_bool_numeric_inputs():
    with pytest.raises(ValueError, match="max_per_class"):
        select_class_limited_indices([1, 2], True)
    with pytest.raises(ValueError, match="seed"):
        normalize_class_limit_seed(True)
    with pytest.raises(ValueError, match="seed"):
        normalize_class_limit_seed(np.bool_(True))
    with pytest.raises(ValueError, match="seed_context"):
        select_class_limited_indices([1, 1, 1], 1, seed_context=True)
    with pytest.raises(ValueError, match="seed_context"):
        select_class_limited_indices([1, 1, 1], 1, seed_context=[0, np.bool_(False)])


def test_class_limit_normalizers_accept_aliases_and_empty_seed():
    assert normalize_class_limit_selection("random") == "random"
    assert normalize_class_limit_selection("first") == "first"
    assert normalize_class_limit_seed(7) == 7
    assert normalize_class_limit_seed("") is None
    assert normalize_class_limit_seed("  ") is None
