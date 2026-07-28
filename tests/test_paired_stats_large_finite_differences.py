from __future__ import annotations

import numpy as np

from neureptrace.paired_stats import sign_flip_p_value


def test_sign_flip_p_value_is_scale_invariant_for_large_finite_differences() -> None:
    reference = np.asarray([1.0, 1.0, 0.1])
    large = np.asarray([1e308, 1e308, 1e307])

    reference_p = sign_flip_p_value(reference, n_permutations=10_000)
    with np.errstate(over="raise", invalid="raise"):
        large_p = sign_flip_p_value(large, n_permutations=10_000)

    assert reference_p == 0.25
    assert large_p == reference_p
