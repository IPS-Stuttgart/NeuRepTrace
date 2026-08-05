from __future__ import annotations

import inspect

import numpy as np
import pytest

from neureptrace.decoding.structured_sequence import (
    decode_sequence_templates,
    decode_unique_class_assignments,
    learn_sequence_templates,
)


def test_unique_assignment_recovers_one_class_per_event() -> None:
    probabilities = np.asarray(
        [
            [0.70, 0.10, 0.10, 0.10],
            [0.55, 0.54, 0.01, 0.01],
            [0.52, 0.01, 0.51, 0.01],
            [0.52, 0.01, 0.01, 0.51],
        ]
    )

    result = decode_unique_class_assignments(probabilities, ["trial-a"] * 4)

    assert result.predictions.tolist() == [0, 1, 2, 3]
    assert result.selected_structures == ((0, 1, 2, 3),)
    assert result.metadata["uses_evaluation_labels"] is False


def test_unique_assignment_preserves_original_row_order_and_labels() -> None:
    probabilities = np.asarray(
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.9, 0.1],
            [0.2, 0.8],
        ]
    )

    result = decode_unique_class_assignments(
        probabilities,
        ["second", "first", "first", "second"],
        class_labels=["index", "middle"],
    )

    assert result.group_ids == ("second", "first")
    assert result.predictions.tolist() == ["middle", "middle", "index", "index"]


def test_unique_assignment_rejects_incomplete_groups() -> None:
    with pytest.raises(ValueError, match="requires exactly 3"):
        decode_unique_class_assignments(np.full((2, 3), 1.0 / 3.0), ["trial", "trial"])


def test_learn_and_decode_sequence_templates_from_calibration_only() -> None:
    calibration_labels = [
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
        1,
        0,
        3,
        2,
        1,
        0,
        3,
        2,
    ]
    calibration_template_ids = ["A"] * 8 + ["B"] * 8
    calibration_positions = [1, 2, 3, 4] * 4
    templates = learn_sequence_templates(
        calibration_labels,
        calibration_template_ids,
        calibration_positions,
    )
    probabilities = np.asarray(
        [
            [0.10, 0.75, 0.10, 0.05],
            [0.70, 0.15, 0.10, 0.05],
            [0.05, 0.05, 0.10, 0.80],
            [0.05, 0.05, 0.80, 0.10],
        ]
    )

    result = decode_sequence_templates(probabilities, ["evaluation-trial"] * 4, [1, 2, 3, 4], templates)

    assert templates.template_ids == ("A", "B")
    assert result.selected_structures == ("B",)
    assert result.predictions.tolist() == [1, 0, 3, 2]
    assert result.metadata["uses_evaluation_labels"] is False


def test_template_learning_rejects_inconsistent_calibration_rows() -> None:
    with pytest.raises(ValueError, match="inconsistent calibration labels"):
        learn_sequence_templates(
            [0, 1, 1, 1, 1],
            ["A", "A", "A", "A", "A"],
            [1, 2, 3, 4, 1],
            require_permutations=False,
        )


def test_template_decoding_requires_each_position_once() -> None:
    templates = learn_sequence_templates([0, 1], ["A", "A"], [1, 2])
    with pytest.raises(ValueError, match="each template position exactly once"):
        decode_sequence_templates(np.asarray([[0.8, 0.2], [0.2, 0.8]]), ["trial", "trial"], [1, 1], templates)


def test_structured_decoders_accept_tuple_group_ids() -> None:
    result = decode_unique_class_assignments(
        np.asarray([[0.9, 0.1], [0.2, 0.8]]),
        [("subject", 1), ("subject", 1)],
    )
    assert result.group_ids == (("subject", 1),)
    assert result.predictions.tolist() == [0, 1]


def test_decision_apis_do_not_accept_evaluation_labels() -> None:
    assert "labels" not in inspect.signature(decode_unique_class_assignments).parameters
    assert "labels" not in inspect.signature(decode_sequence_templates).parameters
