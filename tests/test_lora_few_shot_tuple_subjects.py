from __future__ import annotations

import numpy as np
import pytest

import neureptrace  # noqa: F401  # installs runtime compatibility patches
from neureptrace.decoding import lora_few_shot


def _tuple_subject_vector() -> np.ndarray:
    subjects = np.empty(8, dtype=object)
    subjects[:] = [
        ("site-a", "subject-01"),
        ("site-a", "subject-01"),
        ("site-a", "subject-01"),
        ("site-a", "subject-01"),
        ("site-b", "subject-02"),
        ("site-b", "subject-02"),
        ("site-b", "subject-02"),
        ("site-b", "subject-02"),
    ]
    return subjects


@pytest.mark.parametrize(
    "subject_key",
    [
        ("site-a", "subject-01"),
        np.asarray(("site-a", "subject-01"), dtype=object),
    ],
)
def test_lora_balanced_episode_selection_treats_tuple_subjects_atomically(subject_key: object) -> None:
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    subjects = _tuple_subject_vector()

    support, query = lora_few_shot._balanced_subject_episode_indices(
        y,
        subjects,
        subject_key,
        support_per_class=1,
        query_per_class=1,
        seed=13,
    )

    assert support.size == 2
    assert query.size == 2
    assert set(y[support].tolist()) == {0, 1}
    assert set(y[query].tolist()) == {0, 1}
    assert set(support.tolist()).isdisjoint(query.tolist())
    selected_subjects = [subjects[int(index)] for index in np.concatenate([support, query])]
    assert selected_subjects == [("site-a", "subject-01")] * 4


def test_lora_balanced_episode_selection_returns_empty_for_insufficient_tuple_subject_rows() -> None:
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    subjects = _tuple_subject_vector()

    support, query = lora_few_shot._balanced_subject_episode_indices(
        y,
        subjects,
        ("site-a", "subject-01"),
        support_per_class=2,
        query_per_class=1,
        seed=13,
    )

    assert support.size == 0
    assert query.size == 0
