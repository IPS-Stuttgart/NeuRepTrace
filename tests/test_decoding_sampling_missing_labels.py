from __future__ import annotations

import numpy as np

from neureptrace.decoding.sampling import select_class_limited_indices


def test_class_limit_first_collapses_equivalent_missing_labels() -> None:
    labels = [
        ("face", float("nan")),
        ("face", np.float64("nan")),
        ("house", "left"),
        ("house", "left"),
    ]

    selected = select_class_limited_indices(labels, max_per_class=1, selection="first")

    assert selected.tolist() == [0, 2]


def test_class_limit_random_retains_one_missing_label_row() -> None:
    labels = [float("nan"), np.float64("nan"), "house", "house"]

    selected = select_class_limited_indices(labels, max_per_class=1, selection="random", seed=7)

    assert selected.shape == (2,)
    assert np.count_nonzero(selected < 2) == 1
    assert np.count_nonzero(selected >= 2) == 1
