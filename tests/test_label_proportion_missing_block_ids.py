from __future__ import annotations

import numpy as np

from neureptrace.decoding.label_proportions import adjust_probability_blocks_to_label_proportions


def _probabilities() -> np.ndarray:
    return np.asarray(
        [
            [0.80, 0.20],
            [0.70, 0.30],
            [0.30, 0.70],
            [0.20, 0.80],
        ]
    )


def test_label_proportion_blocks_group_and_lookup_distinct_nan_ids() -> None:
    missing_rows = [float("nan"), float("nan")]
    missing_key = float("nan")

    result = adjust_probability_blocks_to_label_proportions(
        _probabilities(),
        [*missing_rows, "run2", "run2"],
        {
            missing_key: [0.75, 0.25],
            "run2": [0.25, 0.75],
        },
        tol=1e-10,
    )

    assert result.metadata["n_blocks"] == 2
    assert tuple(row["n_samples"] for row in result.block_metadata) == (2, 2)
    np.testing.assert_allclose(result.probabilities[:2].mean(axis=0), [0.75, 0.25], atol=1e-8)
    np.testing.assert_allclose(result.probabilities[2:].mean(axis=0), [0.25, 0.75], atol=1e-8)


def test_label_proportion_blocks_match_nan_inside_tuple_ids() -> None:
    missing_rows = [("s01", float("nan")), ("s01", float("nan"))]
    missing_key = ("s01", float("nan"))

    result = adjust_probability_blocks_to_label_proportions(
        _probabilities(),
        [*missing_rows, ("s02", "run2"), ("s02", "run2")],
        {
            missing_key: [0.75, 0.25],
            ("s02", "run2"): [0.25, 0.75],
        },
        tol=1e-10,
    )

    assert result.metadata["n_blocks"] == 2
    assert tuple(row["n_samples"] for row in result.block_metadata) == (2, 2)
    np.testing.assert_allclose(result.probabilities[:2].mean(axis=0), [0.75, 0.25], atol=1e-8)
    np.testing.assert_allclose(result.probabilities[2:].mean(axis=0), [0.25, 0.75], atol=1e-8)
