from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_balance import (
    SourceBalanceConfig,
    compute_source_balance_weights,
    resample_source_rows_balanced,
)


def test_direct_source_balance_config_is_revalidated() -> None:
    config = SourceBalanceConfig(strategy="labels", target="undersample", normalize_weights="false", random_state="7")  # type: ignore[arg-type]

    result = compute_source_balance_weights(["a", "a", "b"], config=config)

    assert result.metadata["source_balance_strategy"] == "class"
    assert result.metadata["source_balance_target"] == "min"
    assert result.metadata["source_balance_normalize_weights"] is False
    assert np.allclose(result.sample_weights, [0.5, 0.5, 1.0])


@pytest.mark.parametrize("bad_normalize_weights", [np.asarray(True), np.asarray([True])])
def test_direct_source_balance_config_rejects_array_normalize_weights(bad_normalize_weights: np.ndarray) -> None:
    config = SourceBalanceConfig(normalize_weights=bad_normalize_weights)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="normalize_weights"):
        compute_source_balance_weights(["a", "b"], config=config)


def test_direct_source_balance_config_rejects_array_random_state() -> None:
    config = SourceBalanceConfig(random_state=np.asarray([4]))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="random_state"):
        resample_source_rows_balanced(np.asarray([[0.0], [1.0]], dtype=float), ["a", "b"], config=config)
