import numpy as np
import pytest

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


def test_label_proportion_blocks_accept_single_column_block_vector():
    blocks = np.asarray([["run1"], ["run1"], ["run2"], ["run2"]], dtype=object)

    result = adjust_probability_blocks_to_label_proportions(
        _probabilities(),
        blocks,
        {
            "run1": [0.75, 0.25],
            "run2": [0.25, 0.75],
        },
        tol=1e-10,
    )

    flat_blocks = blocks.reshape(-1)
    assert result.metadata["n_blocks"] == 2
    assert np.allclose(result.probabilities[flat_blocks == "run1"].mean(axis=0), [0.75, 0.25], atol=1e-8)
    assert np.allclose(result.probabilities[flat_blocks == "run2"].mean(axis=0), [0.25, 0.75], atol=1e-8)


def test_label_proportion_blocks_preserve_tuple_block_ids():
    blocks = [("s01", "run1"), ("s01", "run1"), ("s02", "run2"), ("s02", "run2")]

    result = adjust_probability_blocks_to_label_proportions(
        _probabilities(),
        blocks,
        {
            ("s01", "run1"): [0.75, 0.25],
            ("s02", "run2"): [0.25, 0.75],
        },
        tol=1e-10,
    )

    assert result.metadata["n_blocks"] == 2
    assert tuple(row["block"] for row in result.block_metadata) == ("('s01', 'run1')", "('s02', 'run2')")


def test_label_proportion_blocks_reject_matrix_shaped_block_ids():
    malformed_blocks = np.asarray([["run1", "run1"], ["run2", "run2"]], dtype=object)

    with pytest.raises(ValueError, match="block_ids.*same row count"):
        adjust_probability_blocks_to_label_proportions(
            _probabilities(),
            malformed_blocks,
            {
                "run1": [0.75, 0.25],
                "run2": [0.25, 0.75],
            },
        )


def test_label_proportion_blocks_reject_nested_unhashable_block_ids():
    malformed_blocks = [["run1"], ["run1"], ["run2"], ["run2"]]

    with pytest.raises(ValueError, match="hashable block identifiers"):
        adjust_probability_blocks_to_label_proportions(
            _probabilities(),
            malformed_blocks,
            {
                "run1": [0.75, 0.25],
                "run2": [0.25, 0.75],
            },
        )
