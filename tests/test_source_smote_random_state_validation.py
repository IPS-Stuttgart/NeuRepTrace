from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_smote import source_smote_config


def test_source_smote_config_accepts_none_like_random_state_tokens() -> None:
    for value in [None, "", " none ", "NULL"]:
        assert source_smote_config(random_state=value).random_state is None


def test_source_smote_config_rejects_non_scalar_random_state_values() -> None:
    for value in ([1], {"seed": 1}, np.asarray([1])):
        with pytest.raises(ValueError, match="random_state"):
            source_smote_config(random_state=value)  # type: ignore[arg-type]
