from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_smote import source_smote_config


def test_source_smote_config_accepts_none_like_random_state_tokens() -> None:
    for value in [None, "", " none ", "NULL", np.asarray("none")]:
        assert source_smote_config(random_state=value).random_state is None


def test_source_smote_config_accepts_scalar_array_random_state() -> None:
    assert source_smote_config(random_state=np.asarray(7)).random_state == 7


def test_source_smote_config_rejects_non_scalar_random_state_values() -> None:
    for value in ([1], {"value": 1}, np.asarray([1])):
        with pytest.raises(ValueError, match="random_state"):
            source_smote_config(random_state=value)  # type: ignore[arg-type]


def test_source_smote_core_config_normalizes_random_state_without_compatibility_patch() -> None:
    core_config = getattr(source_smote_config, "__wrapped__", source_smote_config)

    for value in [None, "", " none ", "NULL", np.asarray("none")]:
        assert core_config(random_state=value).random_state is None

    assert core_config(random_state=np.asarray(7)).random_state == 7

    for value in ([1], {"value": 1}, np.asarray([1])):
        with pytest.raises(ValueError, match="random_state"):
            core_config(random_state=value)  # type: ignore[arg-type]
