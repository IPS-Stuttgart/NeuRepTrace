from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding import source_balance


@pytest.mark.parametrize("random_state", [2**53 + 1, str(2**53 + 1)])
def test_source_balance_preserves_exact_large_random_state(monkeypatch, random_state: int | str) -> None:
    captured_seeds: list[int] = []

    def fake_default_rng(seed: int) -> object:
        captured_seeds.append(seed)
        return object()

    monkeypatch.setattr(source_balance.np.random, "default_rng", fake_default_rng)

    result = source_balance.resample_source_rows_balanced(
        np.asarray([[0.0], [1.0]], dtype=float),
        ["left", "right"],
        config={"strategy": "none", "random_state": random_state},
    )

    assert result.source_indices.tolist() == [0, 1]
    assert captured_seeds == [2**53 + 1]
