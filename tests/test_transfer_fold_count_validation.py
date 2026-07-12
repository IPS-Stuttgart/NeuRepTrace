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
