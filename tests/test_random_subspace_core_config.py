from __future__ import annotations

import importlib

import numpy as np
import pytest

import neureptrace.decoding.random_subspace as random_subspace
from neureptrace import _subspace_bool_config_patch


@pytest.fixture()
def core_random_subspace():
    module = importlib.reload(random_subspace)
    try:
        yield module
    finally:
        importlib.reload(random_subspace)
        _subspace_bool_config_patch.install()


def test_core_random_subspace_config_parses_boolean_values(core_random_subspace) -> None:
    assert core_random_subspace.random_subspace_ensemble_config(bootstrap_rows="false").bootstrap_rows is False
    assert core_random_subspace.random_subspace_ensemble_config(bootstrap_rows="OFF").bootstrap_rows is False
    assert core_random_subspace.random_subspace_ensemble_config(bootstrap_rows="yes").bootstrap_rows is True
    assert core_random_subspace.random_subspace_ensemble_config(bootstrap_rows=np.bool_(False)).bootstrap_rows is False
    assert core_random_subspace.random_subspace_ensemble_config(bootstrap_rows=1).bootstrap_rows is True


@pytest.mark.parametrize("value", ["maybe", 2, -1, 0.5, np.asarray([False, True])])
def test_core_random_subspace_config_rejects_ambiguous_boolean_values(core_random_subspace, value) -> None:
    with pytest.raises(ValueError, match="bootstrap_rows"):
        core_random_subspace.random_subspace_ensemble_config(bootstrap_rows=value)


def test_core_random_subspace_config_parses_optional_random_state_values(core_random_subspace) -> None:
    assert core_random_subspace.random_subspace_ensemble_config(random_state=" none ").random_state is None
    assert core_random_subspace.random_subspace_ensemble_config(random_state=np.asarray("NULL", dtype=object)).random_state is None
    assert core_random_subspace.random_subspace_ensemble_config(random_state="7").random_state == 7

    first = core_random_subspace.sample_feature_subspaces(n_features=4, n_estimators=2, random_state=" none ")
    second = core_random_subspace.sample_feature_subspaces(n_features=4, n_estimators=2, random_state=np.asarray("null", dtype=object))
    assert len(first) == 2
    assert len(second) == 2


@pytest.mark.parametrize("value", [True, -1, 0.5, [], {}, np.asarray([1, 2])])
def test_core_random_subspace_config_rejects_invalid_random_state_values(core_random_subspace, value) -> None:
    with pytest.raises(ValueError, match="random_state"):
        core_random_subspace.random_subspace_ensemble_config(random_state=value)
    with pytest.raises(ValueError, match="random_state"):
        core_random_subspace.sample_feature_subspaces(n_features=4, random_state=value)
