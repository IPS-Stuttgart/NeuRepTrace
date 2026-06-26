import numpy as np
import pytest

from neureptrace.decoding.label_proportions import (
    WEAK_LABEL_PROPORTION_CATEGORY,
    adjust_probabilities_to_label_proportions,
    adjust_probability_blocks_to_label_proportions,
    normalize_label_proportions,
    predict_labels_from_label_proportions,
)


def test_adjust_probabilities_to_label_proportions_matches_requested_target_prior():
    probabilities = np.asarray(
        [
            [0.90, 0.10],
            [0.70, 0.30],
            [0.60, 0.40],
            [0.40, 0.60],
            [0.30, 0.70],
            [0.20, 0.80],
        ]
    )

    result = adjust_probabilities_to_label_proportions(
        probabilities,
        {"target": 1, "non_target": 3},
        classes=("target", "non_target"),
        tol=1e-10,
    )

    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert np.allclose(result.probabilities.mean(axis=0), [0.25, 0.75], atol=1e-8)
    assert result.classes == ("target", "non_target")
    assert result.converged
    assert result.metadata["protocol_category"] == WEAK_LABEL_PROPORTION_CATEGORY
    assert result.metadata["uses_target_trial_labels"] is False
    assert result.metadata["uses_target_label_proportions"] is True
    assert result.metadata["valid_for_strict_source_only"] is False


def test_adjust_probability_blocks_to_label_proportions_matches_each_block_prior():
    probabilities = np.asarray(
        [
            [0.80, 0.20],
            [0.70, 0.30],
            [0.20, 0.80],
            [0.10, 0.90],
            [0.65, 0.35],
            [0.55, 0.45],
            [0.45, 0.55],
            [0.35, 0.65],
        ]
    )
    blocks = np.asarray(["run1", "run1", "run1", "run1", "run2", "run2", "run2", "run2"], dtype=object)

    result = adjust_probability_blocks_to_label_proportions(
        probabilities,
        blocks,
        {
            "run1": [0.25, 0.75],
            "run2": [0.75, 0.25],
        },
        classes=("rare", "standard"),
        tol=1e-10,
    )

    assert result.metadata["blockwise"] is True
    assert result.metadata["n_blocks"] == 2
    assert len(result.block_metadata) == 2
    assert np.allclose(result.probabilities[blocks == "run1"].mean(axis=0), [0.25, 0.75], atol=1e-8)
    assert np.allclose(result.probabilities[blocks == "run2"].mean(axis=0), [0.75, 0.25], atol=1e-8)
    assert set(predict_labels_from_label_proportions(result)).issubset({"rare", "standard"})


def test_adjust_probability_blocks_to_label_proportions_preserves_tuple_block_ids():
    probabilities = np.asarray(
        [
            [0.85, 0.15],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.15, 0.85],
        ]
    )
    blocks = [("subject1", "run1"), ("subject1", "run1"), ("subject1", "run2"), ("subject1", "run2")]

    result = adjust_probability_blocks_to_label_proportions(
        probabilities,
        blocks,
        {
            ("subject1", "run1"): [0.75, 0.25],
            ("subject1", "run2"): [0.25, 0.75],
        },
        classes=("rare", "standard"),
        tol=1e-10,
    )

    run1_mask = np.asarray([block == ("subject1", "run1") for block in blocks], dtype=bool)
    run2_mask = np.asarray([block == ("subject1", "run2") for block in blocks], dtype=bool)
    assert result.metadata["n_blocks"] == 2
    assert np.allclose(result.probabilities[run1_mask].mean(axis=0), [0.75, 0.25], atol=1e-8)
    assert np.allclose(result.probabilities[run2_mask].mean(axis=0), [0.25, 0.75], atol=1e-8)


def test_adjust_probability_blocks_to_label_proportions_preserves_matrix_composite_block_ids():
    probabilities = np.asarray(
        [
            [0.85, 0.15],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.15, 0.85],
        ]
    )
    blocks = np.asarray(
        [
            ["subject1", "run1"],
            ["subject1", "run1"],
            ["subject1", "run2"],
            ["subject1", "run2"],
        ],
        dtype=object,
    )

    result = adjust_probability_blocks_to_label_proportions(
        probabilities,
        blocks,
        {
            ("subject1", "run1"): [0.75, 0.25],
            ("subject1", "run2"): [0.25, 0.75],
        },
        classes=("rare", "standard"),
        tol=1e-10,
    )

    run1_mask = np.asarray([tuple(row) == ("subject1", "run1") for row in blocks], dtype=bool)
    run2_mask = np.asarray([tuple(row) == ("subject1", "run2") for row in blocks], dtype=bool)
    assert result.metadata["n_blocks"] == 2
    assert np.allclose(result.probabilities[run1_mask].mean(axis=0), [0.75, 0.25], atol=1e-8)
    assert np.allclose(result.probabilities[run2_mask].mean(axis=0), [0.25, 0.75], atol=1e-8)


def test_adjust_probability_blocks_to_label_proportions_preserves_square_matrix_composite_block_ids():
    probabilities = np.asarray(
        [
            [0.85, 0.15],
            [0.15, 0.85],
        ]
    )
    blocks = np.asarray(
        [
            ["subject1", "run1"],
            ["subject2", "run1"],
        ],
        dtype=object,
    )

    result = adjust_probability_blocks_to_label_proportions(
        probabilities,
        blocks,
        {
            ("subject1", "run1"): [1.0, 0.0],
            ("subject2", "run1"): [0.0, 1.0],
        },
        classes=("rare", "standard"),
        tol=1e-10,
    )

    assert result.metadata["n_blocks"] == 2
    assert np.allclose(result.probabilities[0], [1.0, 0.0], atol=1e-8)
    assert np.allclose(result.probabilities[1], [0.0, 1.0], atol=1e-8)


def test_adjust_probability_blocks_to_label_proportions_preserves_single_row_composite_block_id():
    probabilities = np.asarray([[0.65, 0.35]])
    blocks = np.asarray([["subject1", "run1"]], dtype=object)

    result = adjust_probability_blocks_to_label_proportions(
        probabilities,
        blocks,
        {("subject1", "run1"): [1.0, 0.0]},
        classes=("rare", "standard"),
        tol=1e-10,
    )

    assert result.metadata["n_blocks"] == 1
    assert np.allclose(result.probabilities[0], [1.0, 0.0], atol=1e-8)


def test_adjust_probability_blocks_to_label_proportions_preserves_row_vector_block_ids():
    probabilities = np.asarray(
        [
            [0.85, 0.15],
            [0.15, 0.85],
        ]
    )
    blocks = np.asarray([["run1", "run2"]], dtype=object)

    result = adjust_probability_blocks_to_label_proportions(
        probabilities,
        blocks,
        {
            "run1": [1.0, 0.0],
            "run2": [0.0, 1.0],
        },
        classes=("rare", "standard"),
        tol=1e-10,
    )

    assert result.metadata["n_blocks"] == 2
    assert np.allclose(result.probabilities[0], [1.0, 0.0], atol=1e-8)
    assert np.allclose(result.probabilities[1], [0.0, 1.0], atol=1e-8)


def test_adjust_probabilities_to_label_proportions_handles_epsilon_only_active_support():
    result = adjust_probabilities_to_label_proportions(
        np.asarray(
            [
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=float,
        ),
        [1.0, 0.0],
        classes=("rare", "standard"),
        tol=1e-12,
    )

    assert result.converged
    assert result.classes == ("rare", "standard")
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert np.allclose(result.probabilities, [[1.0, 0.0], [1.0, 0.0]])
    assert np.allclose(result.probabilities.mean(axis=0), [1.0, 0.0])


def test_predict_labels_from_label_proportions_preserves_tuple_classes_as_atomic_labels():
    result = adjust_probabilities_to_label_proportions(
        [[0.99, 0.01], [0.01, 0.99]],
        {(0, "left"): 1, (1, "right"): 1},
        classes=((0, "left"), (1, "right")),
        tol=1e-12,
    )

    predictions = predict_labels_from_label_proportions(result)

    assert predictions.shape == (2,)
    assert predictions.dtype == object
    assert list(predictions) == [(0, "left"), (1, "right")]


def test_normalize_label_proportions_accepts_counts_and_preserves_class_order():
    proportions, classes = normalize_label_proportions({"standard": 9, "target": 1}, classes=("target", "standard"))

    assert classes == ("target", "standard")
    assert np.allclose(proportions, [0.1, 0.9])


def test_label_proportion_calibration_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="same number of classes"):
        adjust_probabilities_to_label_proportions([[0.6, 0.4]], [0.2, 0.3, 0.5])

    with pytest.raises(ValueError, match="finite and non-negative"):
        normalize_label_proportions([0.5, -0.5])

    with pytest.raises(KeyError, match="Missing target label proportions"):
        adjust_probability_blocks_to_label_proportions(
            [[0.6, 0.4], [0.4, 0.6]],
            ["a", "b"],
            {"a": [0.5, 0.5]},
        )
