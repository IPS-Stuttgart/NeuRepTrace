from __future__ import annotations

import numpy as np
import pytest

from neureptrace import inference


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_t_statistic_rejects_non_finite_subject_effects(invalid_value: float) -> None:
    effects = np.asarray(
        [
            [0.2, invalid_value],
            [0.4, 1.0],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="Subject-level effects must contain only finite values"):
        inference._t_statistic(effects)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_sign_flip_statistics_reject_non_finite_subject_effects(invalid_value: float) -> None:
    effects = np.asarray(
        [
            [0.2, invalid_value],
            [0.4, 1.0],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="Subject-level effects must contain only finite values"):
        inference._sign_flip_t_statistics(
            effects,
            n_permutations=17,
            random_state=13,
        )
