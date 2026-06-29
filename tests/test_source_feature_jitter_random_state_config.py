from __future__ import annotations

import numpy as np
import pytest

import neureptrace.decoding.source_jitter as source_jitter
from neureptrace.decoding.source_jitter import source_feature_jitter_config


@pytest.mark.parametrize("value", [None, "", " none ", "NULL", np.asarray("none")])
def test_random_state_accepts_none_like_values(value: object) -> None:
    config = source_feature_jitter_config(random_state=value)

    assert config.random_state is None


@pytest.mark.parametrize("value", [3, "3", np.asarray(3)])
def test_random_state_accepts_integer_values(value: object) -> None:
    config = source_feature_jitter_config(random_state=value)

    assert config.random_state == 3


def test_random_state_core_helper_accepts_scalar_arrays() -> None:
    assert source_jitter._optional_nonnegative_int(np.asarray("none"), name="random_state") is None
    assert source_jitter._optional_nonnegative_int(np.asarray(3), name="random_state") == 3


@pytest.mark.parametrize("value", [True, -1, 0.5, "1.5", [1], np.asarray([1, 2])])
def test_source_feature_jitter_rejects_invalid_random_state(value: object) -> None:
    with pytest.raises(ValueError, match="random_state"):
        source_feature_jitter_config(random_state=value)


def test_random_state_core_helper_rejects_non_scalar_values() -> None:
    for value in ([1], np.asarray([1, 2])):
        with pytest.raises(ValueError, match="random_state"):
            source_jitter._optional_nonnegative_int(value, name="random_state")
