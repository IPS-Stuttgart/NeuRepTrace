from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.random_subspace import random_subspace_ensemble_config, sample_feature_subspaces


@pytest.mark.parametrize("value", [None, "", " none ", "NULL", np.asarray("none")])
def test_random_subspace_random_state_accepts_none_like_values(value) -> None:
    assert random_subspace_ensemble_config(random_state=value).random_state is None


@pytest.mark.parametrize("value", [7, "7", np.asarray(7)])
def test_random_subspace_random_state_accepts_integer_seeds(value) -> None:
    assert random_subspace_ensemble_config(random_state=value).random_state == 7
    first = sample_feature_subspaces(n_features=5, n_estimators=2, random_state=value)
    second = sample_feature_subspaces(n_features=5, n_estimators=2, random_state=7)
    assert [subset.tolist() for subset in first] == [subset.tolist() for subset in second]


@pytest.mark.parametrize("value", [True, -1, 0.5, "1.5", [1], {"seed": 1}, np.asarray([1, 2])])
def test_random_subspace_random_state_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError, match="random_state"):
        random_subspace_ensemble_config(random_state=value)
    with pytest.raises(ValueError, match="random_state"):
        sample_feature_subspaces(n_features=5, n_estimators=2, random_state=value)
