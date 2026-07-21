from __future__ import annotations

import pytest

from neureptrace import temporal_model


@pytest.mark.parametrize("random_seed", [2**53 + 1, str(2**53 + 1)])
def test_temporal_control_preserves_exact_large_random_seed(monkeypatch, random_seed: int | str) -> None:
    captured_seeds: list[int] = []

    def fake_default_rng(seed: int) -> object:
        captured_seeds.append(seed)
        return object()

    monkeypatch.setattr(temporal_model.np.random, "default_rng", fake_default_rng)

    fits = temporal_model._fit_control(
        [],
        control="shuffled_time",
        n_permutations=0,
        random_seed=random_seed,
        stay_grid_size=2,
    )

    assert fits == []
    assert captured_seeds == [2**53 + 1]
