from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.transfer import cross_validate_feature_decoding, sequential_fold_ids


def test_sequential_fold_ids_rejects_more_folds_than_trials() -> None:
    with pytest.raises(ValueError, match="n_folds must not exceed n_trials"):
        sequential_fold_ids(2, 3)


def test_feature_cross_validation_rejects_more_folds_than_trials() -> None:
    with pytest.raises(ValueError, match="n_folds must not exceed n_trials"):
        cross_validate_feature_decoding(
            np.asarray([[-1.0], [1.0]]),
            np.asarray([0, 1]),
            n_folds=3,
            components_pca=float("inf"),
        )


@pytest.mark.parametrize(
    ("n_trials", "n_folds", "name"),
    [
        (True, 1, "n_trials"),
        (4, False, "n_folds"),
        (2.5, 1, "n_trials"),
        (4, 1.5, "n_folds"),
        (np.asarray(4), 1, "n_trials"),
        (4, np.asarray(2), "n_folds"),
        (4, 2 + 0j, "n_folds"),
    ],
)
def test_sequential_fold_ids_rejects_non_integer_controls(n_trials: object, n_folds: object, name: str) -> None:
    with pytest.raises(ValueError, match=rf"{name} must be a positive integer"):
        sequential_fold_ids(n_trials, n_folds)  # type: ignore[arg-type]


def test_sequential_fold_ids_accepts_numpy_integer_controls() -> None:
    assert sequential_fold_ids(np.int64(4), np.int64(2)).tolist() == [1, 1, 2, 2]


def test_feature_cross_validation_rejects_single_fold() -> None:
    with pytest.raises(ValueError, match="n_folds must be at least 2 for cross-validation"):
        cross_validate_feature_decoding(
            np.asarray([[-1.0], [1.0]]),
            np.asarray([0, 1]),
            n_folds=1,
            components_pca=float("inf"),
        )


def test_feature_cross_validation_rejects_fractional_fold_count() -> None:
    with pytest.raises(ValueError, match="n_folds must be a positive integer"):
        cross_validate_feature_decoding(
            np.asarray([[-1.0], [1.0]]),
            np.asarray([0, 1]),
            n_folds=1.5,
            components_pca=float("inf"),
        )
