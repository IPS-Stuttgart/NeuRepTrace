from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_balance import source_balance_config


def test_source_balance_config_accepts_none_like_random_state_tokens() -> None:
    assert source_balance_config(random_state=None).random_state is None
    assert source_balance_config(random_state="").random_state is None
    assert source_balance_config(random_state=" none ").random_state is None
    assert source_balance_config(random_state="NULL").random_state is None


@pytest.mark.parametrize("random_state", [True, [1], (1,), {"seed": 1}, np.asarray([1])])
def test_source_balance_config_rejects_non_scalar_random_state(random_state: object) -> None:
    with pytest.raises(ValueError, match="random_state"):
        source_balance_config(random_state=random_state)  # type: ignore[arg-type]
