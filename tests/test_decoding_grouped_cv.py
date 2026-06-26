from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding import make_cross_validator, make_tuning_cross_validator


def test_grouped_cross_validator_rejects_class_confined_to_one_group() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    groups = np.array(["g0", "g0", "g0", "g0", "g1", "g2"])

    with pytest.raises(ValueError, match="at least two groups"):
        list(make_cross_validator(labels, groups, n_splits=2))


def test_grouped_tuning_cross_validator_rejects_class_confined_to_one_group() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    groups = np.array(["g0", "g0", "g0", "g0", "g1", "g2"])

    with pytest.raises(ValueError, match="at least two groups"):
        make_tuning_cross_validator(labels, groups, n_splits=3)


def test_grouped_tuning_cross_validator_caps_splits_to_class_group_coverage() -> None:
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    groups = np.array(["g0", "g0", "g0", "g0", "g1", "g1", "g2", "g2", "g3", "g3"])

    splits = make_tuning_cross_validator(labels, groups, n_splits=5)

    assert len(splits) == 4
    for train_idx, test_idx in splits:
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
        assert set(np.unique(labels[train_idx])) == {0, 1}
