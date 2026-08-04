from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding import make_cross_validator, make_tuning_cross_validator


def _labels() -> list[tuple[str, int]]:
    return [
        ("face", 1),
        ("object", 2),
        ("face", 1),
        ("object", 2),
        ("face", 1),
        ("object", 2),
        ("face", 1),
        ("object", 2),
    ]


def _groups() -> list[tuple[str, str]]:
    return [
        ("subject-0", "run-0"),
        ("subject-0", "run-0"),
        ("subject-1", "run-0"),
        ("subject-1", "run-0"),
        ("subject-2", "run-0"),
        ("subject-2", "run-0"),
        ("subject-3", "run-0"),
        ("subject-3", "run-0"),
    ]


def test_ungrouped_cross_validator_treats_tuple_labels_as_atomic_classes() -> None:
    labels = _labels()

    splits = list(make_cross_validator(labels, None, n_splits=2))

    assert len(splits) == 2
    for train_idx, test_idx in splits:
        assert {labels[index] for index in train_idx} == {("face", 1), ("object", 2)}
        assert {labels[index] for index in test_idx} == {("face", 1), ("object", 2)}


def test_grouped_cross_validator_accepts_composite_labels_and_groups() -> None:
    labels = _labels()
    groups = _groups()

    splits = list(make_cross_validator(labels, groups, n_splits=2))

    assert len(splits) == 2
    for train_idx, test_idx in splits:
        train_groups = {groups[index] for index in train_idx}
        test_groups = {groups[index] for index in test_idx}
        assert train_groups.isdisjoint(test_groups)
        assert {labels[index] for index in train_idx} == {("face", 1), ("object", 2)}
        assert {labels[index] for index in test_idx} == {("face", 1), ("object", 2)}


def test_grouped_cross_validator_checks_composite_class_group_coverage() -> None:
    labels = [("face", 1), ("face", 1), ("object", 2), ("object", 2)]
    groups = [("subject-0", "run-0"), ("subject-0", "run-0"), ("subject-1", "run-0"), ("subject-2", "run-0")]

    with pytest.raises(ValueError, match="at least two groups"):
        list(make_cross_validator(labels, groups, n_splits=2))


def test_grouped_tuning_caps_splits_for_composite_identifiers() -> None:
    labels = _labels()
    groups = _groups()

    splits = make_tuning_cross_validator(labels, groups, n_splits=5)

    assert len(splits) == 4
    for train_idx, test_idx in splits:
        assert set(np.asarray(train_idx).tolist()).isdisjoint(set(np.asarray(test_idx).tolist()))
        assert {groups[index] for index in train_idx}.isdisjoint({groups[index] for index in test_idx})
